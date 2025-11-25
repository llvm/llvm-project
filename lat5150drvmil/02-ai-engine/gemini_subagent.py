"""
Gemini Subagent
---------------
Advanced subagent leveraging Google Gemini's capabilities for multimodal AI,
function calling, code execution, and grounding with Google Search.

Key Features:
- Multimodal analysis (text, images, videos, audio)
- Long context support (up to 2M tokens)
- Function calling capabilities
- Code execution with Gemini
- Google Search grounding for factual accuracy
- Thinking mode for extended reasoning
- ACE-FCA compliant output compression

Author: LAT5150DRVMIL AI Platform
"""

import json
import logging
from typing import Any, Dict, List, Optional

from base_subagent import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)


class GeminiAgent(BaseSubagent):
    """
    Gemini-powered subagent with multimodal capabilities.

    Capabilities:
    1. Text generation with thinking mode and grounding
    2. Multimodal analysis (images, videos, audio)
    3. Function calling
    4. Code execution
    5. Long context processing (2M tokens)
    """

    def __init__(self, mcp_client):
        """Initialize Gemini subagent with MCP client."""
        super().__init__(
            name="gemini",
            mcp_client=mcp_client,
            server_name="gemini"
        )
        self.capabilities = [
            "text_generation",
            "multimodal_analysis",
            "function_calling",
            "code_execution",
            "grounding"
        ]
        self.sessions: Dict[str, str] = {}
        logger.info("GeminiAgent initialized with capabilities: %s", self.capabilities)

    async def can_handle(self, task: Dict[str, Any]) -> bool:
        """
        Determine if this agent can handle the given task.

        Args:
            task: Task dictionary with 'type' and other fields

        Returns:
            True if agent can handle the task
        """
        task_type = task.get("type", "")

        # Multimodal tasks
        if any(keyword in task_type for keyword in ["image", "video", "audio", "multimodal"]):
            return True

        # Long context tasks (Gemini supports up to 2M tokens)
        if task.get("long_context", False) or task.get("context_length", 0) > 128000:
            return True

        # Function calling tasks
        if "function_call" in task_type or task.get("functions"):
            return True

        # Code execution tasks
        if "code_execution" in task_type or task.get("execute_code"):
            return True

        # Grounding tasks (fact-checking, web search)
        if "grounding" in task_type or task.get("grounding"):
            return True

        # Thinking tasks (extended reasoning)
        if "thinking" in task_type or task.get("thinking_mode"):
            return True

        return False

    async def execute(self, task: Dict[str, Any]) -> SubagentResult:
        """
        Execute a task using Gemini's capabilities.

        Args:
            task: Task dictionary with type, prompt, and parameters

        Returns:
            SubagentResult with response and metadata
        """
        task_type = task.get("type", "text_generation")

        try:
            if task_type == "text_generation" or task_type.startswith("generate"):
                return await self._handle_text_generation(task)
            elif task_type == "multimodal_analysis" or any(x in task_type for x in ["image", "video", "audio"]):
                return await self._handle_multimodal(task)
            elif task_type == "function_calling":
                return await self._handle_function_calling(task)
            elif task_type == "code_execution":
                return await self._handle_code_execution(task)
            elif task_type == "grounding":
                return await self._handle_grounding(task)
            else:
                # Default to text generation
                return await self._handle_text_generation(task)

        except Exception as e:
            logger.error(f"Error executing Gemini task: {e}", exc_info=True)
            return SubagentResult(
                success=False,
                data={"error": str(e)},
                message=f"Gemini execution failed: {str(e)}",
                agent_name=self.name
            )

    async def _handle_text_generation(self, task: Dict[str, Any]) -> SubagentResult:
        """
        Handle text generation with optional thinking mode and grounding.

        Features:
        - Thinking mode for extended reasoning
        - Google Search grounding for factual accuracy
        - Long context support (up to 2M tokens)
        - ACE-FCA output compression
        """
        prompt = task.get("prompt", "")
        session_id = task.get("session_id")
        thinking_mode = task.get("thinking_mode", False)
        grounding = task.get("grounding", False)
        code_execution = task.get("code_execution", False)

        # Build arguments
        args = {
            "prompt": prompt,
            "thinking_mode": thinking_mode,
            "grounding": grounding,
            "code_execution": code_execution
        }

        if session_id:
            args["session_id"] = session_id

        # Add optional generation parameters
        if "temperature" in task:
            args["temperature"] = task["temperature"]
        if "max_tokens" in task:
            args["max_tokens"] = task["max_tokens"]

        # Execute via MCP
        result = await self.call_mcp_tool("gemini_generate", args)

        if not result.get("success"):
            return SubagentResult(
                success=False,
                data=result,
                message="Text generation failed",
                agent_name=self.name
            )

        response_text = result.get("response", "")

        # Apply ACE-FCA output compression if enabled
        if task.get("compress_output", True):
            response_text = self._apply_ace_fca_compression(response_text, task)

        return SubagentResult(
            success=True,
            data={
                "response": response_text,
                "thinking_mode": thinking_mode,
                "grounding": grounding,
                "code_execution": code_execution,
                "compressed": task.get("compress_output", True)
            },
            message="Text generated successfully",
            agent_name=self.name,
            metadata={
                "capability": "text_generation",
                "thinking_mode": thinking_mode,
                "grounding": grounding
            }
        )

    async def _handle_multimodal(self, task: Dict[str, Any]) -> SubagentResult:
        """
        Handle multimodal analysis (images, videos, audio).

        Supports:
        - Image analysis and OCR
        - Video understanding and transcription
        - Audio transcription and analysis
        """
        prompt = task.get("prompt", "")
        media_path = task.get("media_path", "")
        media_type = task.get("media_type", "image")
        session_id = task.get("session_id")

        if not media_path:
            return SubagentResult(
                success=False,
                data={"error": "No media_path provided"},
                message="Media path is required for multimodal analysis",
                agent_name=self.name
            )

        args = {
            "prompt": prompt,
            "media_path": media_path,
            "media_type": media_type
        }

        if session_id:
            args["session_id"] = session_id

        # Execute via MCP
        result = await self.call_mcp_tool("gemini_multimodal", args)

        if not result.get("success"):
            return SubagentResult(
                success=False,
                data=result,
                message=f"Multimodal analysis failed for {media_type}",
                agent_name=self.name
            )

        response_text = result.get("response", "")

        # Apply ACE-FCA compression
        if task.get("compress_output", True):
            response_text = self._apply_ace_fca_compression(response_text, task)

        return SubagentResult(
            success=True,
            data={
                "response": response_text,
                "media_type": media_type,
                "media_path": media_path,
                "compressed": task.get("compress_output", True)
            },
            message=f"{media_type.capitalize()} analysis completed",
            agent_name=self.name,
            metadata={
                "capability": "multimodal_analysis",
                "media_type": media_type
            }
        )

    async def _handle_function_calling(self, task: Dict[str, Any]) -> SubagentResult:
        """
        Handle function calling with Gemini.

        Uses JSON schema to define functions and let Gemini decide when
        and how to call them based on the prompt.
        """
        prompt = task.get("prompt", "")
        functions = task.get("functions", [])
        session_id = task.get("session_id")

        if not functions:
            return SubagentResult(
                success=False,
                data={"error": "No functions provided"},
                message="Functions are required for function calling",
                agent_name=self.name
            )

        args = {
            "prompt": prompt,
            "functions": functions
        }

        if session_id:
            args["session_id"] = session_id

        # Execute via MCP
        result = await self.call_mcp_tool("gemini_function_call", args)

        if not result.get("success"):
            return SubagentResult(
                success=False,
                data=result,
                message="Function calling failed",
                agent_name=self.name
            )

        response_text = result.get("response", "")

        # Apply ACE-FCA compression
        if task.get("compress_output", True):
            response_text = self._apply_ace_fca_compression(response_text, task)

        return SubagentResult(
            success=True,
            data={
                "response": response_text,
                "function_count": len(functions),
                "compressed": task.get("compress_output", True)
            },
            message="Function calling completed",
            agent_name=self.name,
            metadata={
                "capability": "function_calling",
                "function_count": len(functions)
            }
        )

    async def _handle_code_execution(self, task: Dict[str, Any]) -> SubagentResult:
        """
        Handle code generation and execution with Gemini.

        Gemini can generate and execute code in a sandboxed environment,
        returning both the code and execution results.
        """
        prompt = task.get("prompt", "")
        session_id = task.get("session_id")

        args = {
            "prompt": prompt
        }

        if session_id:
            args["session_id"] = session_id

        # Execute via MCP
        result = await self.call_mcp_tool("gemini_code_execute", args)

        if not result.get("success"):
            return SubagentResult(
                success=False,
                data=result,
                message="Code execution failed",
                agent_name=self.name
            )

        response_text = result.get("response", "")

        # Apply ACE-FCA compression
        if task.get("compress_output", True):
            response_text = self._apply_ace_fca_compression(response_text, task)

        return SubagentResult(
            success=True,
            data={
                "response": response_text,
                "compressed": task.get("compress_output", True)
            },
            message="Code execution completed",
            agent_name=self.name,
            metadata={
                "capability": "code_execution"
            }
        )

    async def _handle_grounding(self, task: Dict[str, Any]) -> SubagentResult:
        """
        Handle grounding with Google Search for factual accuracy.

        Automatically routes through text generation with grounding enabled.
        """
        task["grounding"] = True
        return await self._handle_text_generation(task)

    def _apply_ace_fca_compression(self, text: str, task: Dict[str, Any]) -> str:
        """
        Apply ACE-FCA output compression methodology.

        Compresses output to 40-60% of original while preserving key information.
        Uses extractive summarization and semantic filtering.

        Args:
            text: Original response text
            task: Task context for compression hints

        Returns:
            Compressed text maintaining semantic value
        """
        target_ratio = task.get("compression_ratio", 0.5)  # Default 50%

        # Simple compression: extract key sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= 2:
            return text  # Too short to compress

        # Keep first, last, and middle sentences based on ratio
        target_count = max(2, int(len(sentences) * target_ratio))

        # Always keep first and last
        compressed = [sentences[0]]

        # Add middle sentences
        if target_count > 2:
            step = len(sentences) // (target_count - 1)
            for i in range(step, len(sentences) - 1, step):
                if len(compressed) < target_count - 1:
                    compressed.append(sentences[i])

        # Add last sentence
        if len(sentences) > 1:
            compressed.append(sentences[-1])

        return '. '.join(compressed) + '.'

    async def create_session(self, session_id: str, **kwargs) -> SubagentResult:
        """
        Create a new Gemini conversation session.

        Args:
            session_id: Unique session identifier
            **kwargs: Session configuration (thinking_mode, grounding)

        Returns:
            SubagentResult with session creation status
        """
        args = {
            "session_id": session_id,
            "thinking_mode": kwargs.get("thinking_mode", False),
            "grounding": kwargs.get("grounding", False)
        }

        result = await self.call_mcp_tool("gemini_session_new", args)

        if result.get("success"):
            self.sessions[session_id] = "active"

        return SubagentResult(
            success=result.get("success", False),
            data=result,
            message=result.get("message", "Session creation completed"),
            agent_name=self.name
        )

    async def get_session_stats(self, session_id: str) -> SubagentResult:
        """
        Get statistics for a Gemini session.

        Args:
            session_id: Session identifier

        Returns:
            SubagentResult with session statistics
        """
        result = await self.call_mcp_tool("gemini_session_stats", {"session_id": session_id})

        return SubagentResult(
            success=result.get("success", False),
            data=result.get("stats"),
            message="Session stats retrieved",
            agent_name=self.name
        )

    async def clear_session(self, session_id: str) -> SubagentResult:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            SubagentResult with clear status
        """
        result = await self.call_mcp_tool("gemini_session_clear", {"session_id": session_id})

        return SubagentResult(
            success=result.get("success", False),
            data=result,
            message=result.get("message", "Session cleared"),
            agent_name=self.name
        )

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return self.capabilities

    def get_capability_description(self, capability: str) -> str:
        """Get description of a specific capability."""
        descriptions = {
            "text_generation": "Advanced text generation with thinking mode and grounding (2M token context)",
            "multimodal_analysis": "Analyze images, videos, and audio with vision and speech understanding",
            "function_calling": "AI-driven function calling using JSON schema definitions",
            "code_execution": "Generate and execute code in a sandboxed environment",
            "grounding": "Google Search integration for factual, grounded responses"
        }
        return descriptions.get(capability, "Unknown capability")
