import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger


@dataclass
class MatchResult:
    """消息匹配结果"""
    matched: bool = False
    name: str = ""
    keyword: str = ""
    question: str = ""
    truncated: bool = False


class PatternMatcher:
    """消息模式匹配引擎
    
    负责对用户消息进行前缀模式匹配，支持以下功能：
    - 精确/模糊匹配名字+关键词组合前缀
    - 提取匹配后的消息主体（问题内容）
    - 消息有效性过滤（空消息、超长消息等）
    """

    def __init__(
        self,
        names: List[str],
        keywords: List[str],
        fuzzy_match: bool = True,
        max_question_length: int = 500,
    ):
        self.names = names
        self.keywords = keywords
        self.fuzzy_match = fuzzy_match
        self.max_question_length = max_question_length
        self._compiled_patterns: Optional[List[Tuple[re.Pattern, str, str]]] = None
        self._build_patterns()

    def _normalize(self, text: str) -> str:
        """文本标准化：根据模糊匹配设置决定是否忽略大小写和空格"""
        if self.fuzzy_match:
            return text.strip().lower()
        return text.strip()

    def _build_patterns(self) -> None:
        """预编译所有「名字+关键词」的正则表达式，提升运行时匹配效率"""
        patterns = []
        for name in self.names:
            for keyword in self.keywords:
                escaped_name = re.escape(name)
                escaped_keyword = re.escape(keyword)
                if self.fuzzy_match:
                    pattern_str = rf"^{escaped_name}\s*{escaped_keyword}\s*(.+)$"
                    flags = re.IGNORECASE
                else:
                    pattern_str = rf"^{escaped_name}{keyword}(.+)$"
                    flags = 0
                try:
                    compiled = re.compile(pattern_str, flags)
                    patterns.append((compiled, name, keyword))
                except re.error as e:
                    logger.warning(f"[PatternMatcher] 正则编译失败 name={name}, keyword={keyword}: {e}")
        self._compiled_patterns = patterns
        logger.info(
            f"[PatternMatcher] 已编译 {len(patterns)} 条匹配规则 "
            f"(names={self.names}, keywords={self.keywords}, fuzzy={self.fuzzy_match})"
        )

    def reload_patterns(
        self,
        names: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        fuzzy_match: Optional[bool] = None,
        max_question_length: Optional[int] = None,
    ) -> None:
        """热更新匹配配置，无需重建实例"""
        if names is not None:
            self.names = names
        if keywords is not None:
            self.keywords = keywords
        if fuzzy_match is not None:
            self.fuzzy_match = fuzzy_match
        if max_question_length is not None:
            self.max_question_length = max_question_length
        self._build_patterns()

    def match(self, message: str) -> MatchResult:
        """对单条消息执行模式匹配
        
        Args:
            message: 用户发送的原始消息文本
            
        Returns:
            MatchResult 对象，包含匹配状态、命中的名字/关键词、提取的问题
        """
        if not message or not message.strip():
            return MatchResult(matched=False)

        for compiled_pattern, name, keyword in self._compiled_patterns:
            match_obj = compiled_pattern.match(message)
            if match_obj:
                question = match_obj.group(1).strip()
                truncated = False
                if len(question) > self.max_question_length:
                    question = question[: self.max_question_length]
                    truncated = True
                    logger.warning(
                        f"[PatternMatcher] 问题已截断 (原始长度={len(match_obj.group(1))}, "
                        f"限制={self.max_question_length})"
                    )
                if not question:
                    logger.debug("[PatternMatcher] 匹配成功但问题为空，跳过")
                    continue
                logger.debug(
                    f"[PatternMatcher] 命中匹配: name='{name}', "
                    f"keyword='{keyword}', question='{question[:50]}...'"
                )
                return MatchResult(
                    matched=True,
                    name=name,
                    keyword=keyword,
                    question=question,
                    truncated=truncated,
                )
        return MatchResult(matched=False)


@dataclass
class LLMCallResult:
    """LLM 调用结果"""
    success: bool = False
    answer: str = ""
    error: str = ""
    latency_ms: float = 0.0


class LLMService:
    """LLM 调用服务
    
    封装与 AstrBot Provider 的交互逻辑，提供：
    - 可配置的 Provider 选择
    - 完善的错误处理与日志记录
    - 服务降级机制
    - 调用延迟统计
    """

    def __init__(self, context: Context, config: Dict[str, Any]):
        self.context = context
        self.config = config
        self._call_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    @property
    def fallback_message(self) -> str:
        return self.config.get("fallback_message", "抱歉，AI 服务暂时无法响应，请稍后再试~")

    @property
    def system_prompt(self) -> str:
        return self.config.get("system_prompt", "你是一个乐于助人的AI助手。请用简洁、友好的方式回答用户的问题。")

    def _get_provider(self):
        """获取 LLM 提供商实例
        
        优先使用配置文件指定的 provider_id，
        若未指定或找不到，则回退到当前会话默认提供商。
        """
        provider_id = self.config.get("llm_provider_id", "").strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id=provider_id)
            if provider:
                logger.debug(f"[LLMService] 使用指定提供商: {provider_id}")
                return provider
            logger.warning(f"[LLMService] 未找到 ID 为 '{provider_id}' 的提供商，尝试使用默认提供商")
        return None

    async def call(self, question: str, umo: str) -> LLMCallResult:
        """调用 LLM 获取回答
        
        Args:
            question: 用户问题文本
            umo: 统一消息来源标识
            
        Returns:
            LLMCallResult 包含调用结果或错误信息
        """
        start_time = time.perf_counter()
        self._call_count += 1

        try:
            provider = self._get_provider()
            if provider is None:
                provider = self.context.get_using_provider(umo=umo)
            if provider is None:
                self._error_count += 1
                logger.error("[LLMService] 无法获取任何可用的 LLM 提供商")
                return LLMCallResult(
                    success=False,
                    error="无可用的 LLM 服务提供商",
                    answer=self.fallback_message,
                )

            logger.info(f"[LLMService] 正在调用 LLM, 问题: '{question[:80]}...'")
            llm_response = await provider.text_chat(
                prompt=question,
                system_prompt=self.system_prompt,
            )

            latency = (time.perf_counter() - start_time) * 1000
            self._total_latency_ms += latency

            if llm_response and hasattr(llm_response, "completion_text"):
                answer = llm_response.completion_text.strip()
                if not answer:
                    self._error_count += 1
                    logger.warning("[LLMService] LLM 返回了空回答")
                    return LLMCallResult(
                        success=False,
                        error="LLM 返回空回答",
                        answer=self.fallback_message,
                        latency_ms=latency,
                    )
                logger.info(
                    f"[LLMService] LLM 调用成功, 耗时: {latency:.0f}ms, "
                    f"回答长度: {len(answer)}"
                )
                return LLMCallResult(success=True, answer=answer, latency_ms=latency)

            self._error_count += 1
            logger.error(f"[LLMService] LLM 返回异常响应: {type(llm_response)}")
            return LLMCallResult(
                success=False,
                error=f"LLM 返回异常响应类型: {type(llm_response).__name__}",
                answer=self.fallback_message,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            self._error_count += 1
            logger.error(f"[LLMService] LLM 调用异常: {e}", exc_info=True)
            return LLMCallResult(
                success=False,
                error=str(e),
                answer=self.fallback_message,
                latency_ms=latency,
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        avg_latency = (
            self._total_latency_ms / self._call_count if self._call_count > 0 else 0
        )
        return {
            "total_calls": self._call_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._call_count - self._error_count) / self._call_count * 100
                if self._call_count > 0
                else 0
            ),
            "avg_latency_ms": round(avg_latency, 2),
        }


@register("askyou", "慵懒午睡", "基于模式匹配的AI问答插件，支持自定义触发词和名字列表", "2.0.0")
class MyPlugin(Star):
    """基于模式匹配的 AI 问答插件
    
    核心工作流程：
    1. 接收所有消息事件
    2. 使用 PatternMatcher 进行前缀模式匹配（如 "akt认为..."）
    3. 匹配成功后提取问题内容，通过 LLMService 调用 LLM
    4. 将 LLM 回答返回给用户
    """

    def __init__(self, context: Context, config):
        super().__init__(context)
        self.config = config
        self.matcher: Optional[PatternMatcher] = None
        self.llm_service: Optional[LLMService] = None
        self._init_services()

    def _init_services(self) -> None:
        """初始化匹配引擎和 LLM 服务"""
        names = self.config.get("names", ["akt"])
        keywords = self.config.get("trigger_keywords", ["认为"])
        fuzzy = self.config.get("enable_fuzzy_match", True)
        max_len = self.config.get("max_question_length", 500)

        self.matcher = PatternMatcher(
            names=names,
            keywords=keywords,
            fuzzy_match=fuzzy,
            max_question_length=max_len,
        )
        self.llm_service = LLMService(context=self.context, config=dict(self.config))
        logger.info("[MyPlugin] 初始化完成，服务已就绪")

    def _reload_from_config(self) -> None:
        """从当前配置热更新所有子服务（无需重启插件）"""
        if self.matcher:
            self.matcher.reload_patterns(
                names=self.config.get("names"),
                keywords=self.config.get("trigger_keywords"),
                fuzzy_match=self.config.get("enable_fuzzy_match"),
                max_question_length=self.config.get("max_question_length"),
            )
        if self.llm_service:
            self.llm_service.config = dict(self.config)
        logger.info("[MyPlugin] 配置已热更新")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """全局消息监听入口
        
        对每条消息进行模式匹配，命中后调用 LLM 并返回结果。
        未命中的消息不做任何处理，放行给后续插件/AstrBot 主流程。
        """
        try:
            message_str = event.message_str.strip() if event.message_str else ""
            if not message_str:
                return

            match_result = self.matcher.match(message_str)
            if not match_result.matched:
                return

            logger.info(
                f"[MyPlugin] 消息匹配成功: name='{match_result.name}', "
                f"keyword='{match_result.keyword}', question='{match_result.question[:60]}...'"
            )

            if match_result.truncated:
                yield event.plain_result(
                    f"⚠️ 您的问题过长，已自动截断至 {self.config.get('max_question_length', 500)} 字符。"
                )

            umo = event.unified_msg_origin
            llm_result = await self.llm_service.call(
                question=match_result.question, umo=umo
            )

            if llm_result.success:
                yield event.plain_result(llm_result.answer)
            else:
                logger.warning(
                    f"[MyPlugin] LLM 调用失败: {llm_result.error}, 使用降级回复"
                )
                yield event.plain_result(llm_result.answer)

        except Exception as e:
            logger.error(f"[MyPlugin] 消息处理异常: {e}", exc_info=True)
            yield event.plain_result(
                self.config.get(
                    "fallback_message",
                    "抱歉，处理过程中出现异常，请稍后再试~",
                )
            )

    @filter.command("ask_stats")
    async def cmd_stats(self, event: AstrMessageEvent):
        '''查看 LLM 调用统计信息'''
        stats = self.llm_service.get_stats()
        yield event.plain_result(
            f"📊 LLM 调用统计\n"
            f"总调用次数: {stats['total_calls']}\n"
            f"错误次数: {stats['error_count']}\n"
            f"成功率: {stats['success_rate']:.1f}%\n"
            f"平均延迟: {stats['avg_latency_ms']:.0f}ms"
        )

    @filter.command("ask_reload")
    async def cmd_reload(self, event: AstrMessageEvent):
        '''重新加载配置（热更新）'''
        self._reload_from_config()
        current_names = self.config.get("names", [])
        current_keywords = self.config.get("trigger_keywords", [])
        yield event.plain_result(
            f"✅ 配置已重新加载！\n"
            f"当前名字列表: {current_names}\n"
            f"当前关键词: {current_keywords}"
        )

    @filter.command("ask_test")
    async def cmd_test(self, event: AstrMessageEvent, test_msg: str = ""):
        '''测试 LLM 连通性，可选传入测试问题'''
        question = test_msg or "你好，这是一条连通性测试消息。"
        umo = event.unified_msg_origin
        result = await self.llm_service.call(question=question,umo=umo)
        if result.success:
            yield event.plain_result(
                f"✅ LLM 连通正常 (耗时: {result.latency_ms:.0f}ms)\n"
                f"回复: {result.answer[:200]}"
            )
        else:
            yield event.plain_result(
                f"❌ LLM 连接失败: {result.error}\n"
                f"降级回复: {result.answer}"
            )

    async def terminate(self):
        """插件卸载时释放资源并输出统计摘要"""
        if self.llm_service:
            stats = self.llm_service.get_stats()
            logger.info(
                f"[MyPlugin] 插件卸载 - 统计摘要: 总调用={stats['total_calls']}, "
                f"错误={stats['error_count']}, 成功率={stats['success_rate']:.1f}%"
            )
        logger.info("[MyPlugin] 插件已安全卸载")
