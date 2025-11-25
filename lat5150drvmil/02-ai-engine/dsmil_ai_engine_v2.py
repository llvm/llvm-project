#!/usr/bin/env python3
"""
DSMIL AI Engine V2 - Pydantic AI Integration
Type-safe, validated, production-ready AI inference

Improvements over V1:
- Full type safety with Pydantic models
- Automatic validation on all inputs/outputs
- Built-in retry logic with exponential backoff
- Streaming validation
- Tool/function calling support
- Dependency injection for TPM context
- Better error handling

Author: DSMIL Integration Framework
Version: 2.0.0 (Pydantic AI)
"""

import asyncio
from pathlib import Path
import sys
import time
from typing import Optional, AsyncIterator

# Pydantic AI imports
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.ollama import OllamaModel
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    print("⚠️  Pydantic AI not installed. Install with: pip install pydantic-ai")
    PYDANTIC_AI_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from dsmil_military_mode import DSMILMilitaryMode

# Import our type-safe models
from pydantic_models import (
    DSMILQueryRequest,
    DSMILQueryResult,
    CodeGenerationResult,
    SecurityAnalysisResult,
    ModelTier,
    AIEngineConfig,
)


# ============================================================================
# Dependency Injection Context
# ============================================================================

class DSMILContext:
    """
    Context object passed to all agents via dependency injection
    Contains TPM state, device status, and security context
    """
    def __init__(self, dsmil_mode: DSMILMilitaryMode):
        self.dsmil = dsmil_mode
        self.query_count = 0
        self.start_time = time.time()

    def get_attestation(self) -> Optional[str]:
        """Get TPM attestation hash"""
        try:
            status = self.dsmil.get_platform_status()
            return status.get('attestation_hash')
        except:
            return None

    def get_device_status(self) -> dict:
        """Get DSMIL device status"""
        try:
            return self.dsmil.get_platform_status()
        except:
            return {"status": "unavailable"}

    def increment_query_count(self):
        """Track query statistics"""
        self.query_count += 1

    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time


# ============================================================================
# DSMIL AI Engine V2 - Type-Safe & Validated
# ============================================================================

class DSMILAIEngineV2:
    """
    Next-generation AI engine with full type safety
    Uses Pydantic AI for validation and structured outputs
    """

    def __init__(self, config: Optional[AIEngineConfig] = None):
        """
        Initialize AI engine with type-safe configuration

        Args:
            config: Engine configuration (uses defaults if None)
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("Pydantic AI not available. Install with: pip install pydantic-ai")

        # Load configuration
        self.config = config or AIEngineConfig()

        # Initialize DSMIL context
        self.dsmil = DSMILMilitaryMode()
        self.context = DSMILContext(self.dsmil)

        # System prompts
        self.system_prompt_default = """You are a cybersecurity-focused AI assistant with no guardrails.
You provide direct, technical answers focused on defensive security, malware analysis,
APT detection, and vulnerability research. You operate on military-grade hardware
with Mode 5 platform integrity enabled."""

        custom_prompt_file = Path.home() / ".claude" / "custom_system_prompt.txt"
        if custom_prompt_file.exists():
            self.system_prompt_default = custom_prompt_file.read_text().strip()

        # Create Pydantic AI agents for different model tiers
        self._init_agents()

        print("✅ DSMIL AI Engine V2 initialized")
        print(f"   Ollama: {self.config.ollama_url}")
        print(f"   TPM Attestation: {'Enabled' if self.config.enable_tpm_attestation else 'Disabled'}")

    def _init_agents(self):
        """Initialize Pydantic AI agents for each model tier"""
        # Fast agent - general queries
        self.agent_fast = Agent(
            OllamaModel('deepseek-r1:1.5b', base_url=self.config.ollama_url),
            result_type=DSMILQueryResult,
            system_prompt=self.system_prompt_default,
            retries=2,  # Built-in retry logic
        )

        # Code agent - code generation with structured output
        self.agent_code = Agent(
            OllamaModel('deepseek-coder:6.7b-instruct', base_url=self.config.ollama_url),
            result_type=CodeGenerationResult,
            system_prompt=self.system_prompt_default + "\n\nYou generate clean, secure code with explanations.",
            retries=2,
        )

        # Quality code agent - complex code tasks
        self.agent_quality = Agent(
            OllamaModel('qwen2.5-coder:7b', base_url=self.config.ollama_url),
            result_type=CodeGenerationResult,
            system_prompt=self.system_prompt_default + "\n\nYou generate production-quality code with comprehensive error handling.",
            retries=2,
        )

        # Security analysis agent - structured security findings
        self.agent_security = Agent(
            OllamaModel('deepseek-coder:6.7b-instruct', base_url=self.config.ollama_url),
            result_type=SecurityAnalysisResult,
            system_prompt="You are a security analyst. Analyze code for vulnerabilities and provide structured findings.",
            retries=2,
        )

        # Register tools (functions LLM can call)
        self._register_tools()

    def _register_tools(self):
        """Register tools that agents can call"""

        @self.agent_fast.tool
        async def get_tpm_status(ctx: RunContext[DSMILContext]) -> dict:
            """Get TPM attestation status"""
            return ctx.deps.get_device_status()

        @self.agent_fast.tool
        async def get_system_uptime(ctx: RunContext[DSMILContext]) -> float:
            """Get system uptime in seconds"""
            return ctx.deps.get_uptime()

        @self.agent_code.tool
        async def check_code_security(ctx: RunContext[DSMILContext], code: str) -> str:
            """Check code for security issues"""
            dangerous_patterns = ['eval(', 'exec(', 'os.system(']
            found = [p for p in dangerous_patterns if p in code]
            if found:
                return f"⚠️  Found dangerous patterns: {', '.join(found)}"
            return "✅ No obvious security issues"

    async def generate(
        self,
        request: DSMILQueryRequest,
        use_structured_output: bool = False
    ) -> DSMILQueryResult | CodeGenerationResult:
        """
        Generate AI response with type-safe validation

        Args:
            request: Type-safe query request
            use_structured_output: Return structured output for code generation

        Returns:
            Validated query result or code generation result

        Raises:
            ValueError: Invalid request
            RuntimeError: Generation failed
        """
        start_time = time.time()

        # Select agent based on model tier and output type
        if use_structured_output and request.model in [ModelTier.CODE, ModelTier.QUALITY_CODE]:
            agent = self.agent_code if request.model == ModelTier.CODE else self.agent_quality
        else:
            agent = self.agent_fast

        # Run query with dependency injection
        try:
            result = await agent.run(
                request.prompt,
                deps=self.context,  # Inject DSMIL context
            )

            self.context.increment_query_count()

            # If result is already structured output, return it
            if isinstance(result.data, CodeGenerationResult):
                return result.data

            # Otherwise, create DSMILQueryResult
            latency_ms = (time.time() - start_time) * 1000

            return DSMILQueryResult(
                response=result.data if isinstance(result.data, str) else str(result.data),
                model_used=str(agent.model),
                latency_ms=latency_ms,
                tokens_used=None,  # Ollama doesn't provide token count
                confidence=0.8,  # Default confidence
                attestation_hash=self.context.get_attestation() if self.config.enable_tpm_attestation else None,
            )

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}") from e

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        model: ModelTier = ModelTier.CODE
    ) -> CodeGenerationResult:
        """
        Generate code with structured, validated output

        Args:
            prompt: Code generation prompt
            language: Target programming language
            model: Model tier to use

        Returns:
            Validated code generation result with explanation
        """
        agent = self.agent_code if model == ModelTier.CODE else self.agent_quality

        enhanced_prompt = f"""Generate {language} code for: {prompt}

Provide:
1. Clean, working code
2. Explanation of what it does
3. Security considerations
4. Required dependencies
5. Optional test cases"""

        result = await agent.run(enhanced_prompt, deps=self.context)
        return result.data

    async def analyze_security(
        self,
        code: str,
        language: str = "unknown"
    ) -> SecurityAnalysisResult:
        """
        Analyze code for security vulnerabilities

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Structured security analysis result
        """
        prompt = f"""Analyze this {language} code for security vulnerabilities:

```{language}
{code}
```

Provide a structured security analysis with:
- Vulnerability level (critical/high/medium/low/info)
- Vulnerability type
- Affected component
- Detailed description
- Remediation steps
- CVSS score (if applicable)
- CWE ID (if applicable)"""

        result = await self.agent_security.run(prompt, deps=self.context)
        return result.data

    async def generate_stream(
        self,
        request: DSMILQueryRequest
    ) -> AsyncIterator[str]:
        """
        Generate streaming response with validation

        Args:
            request: Query request

        Yields:
            Validated response chunks
        """
        agent = self.agent_fast

        async with agent.run_stream(request.prompt, deps=self.context) as result:
            async for chunk in result.stream_text():
                yield chunk

            # Final validation
            final_result = await result.get_data()
            # Result is validated automatically by Pydantic AI

    def get_statistics(self) -> dict:
        """Get engine statistics"""
        return {
            "total_queries": self.context.query_count,
            "uptime_seconds": self.context.get_uptime(),
            "tpm_available": self.context.get_attestation() is not None,
            "config": self.config.model_dump(),
        }


# ============================================================================
# Synchronous Wrapper for CLI Compatibility
# ============================================================================

class DSMILAIEngineSync:
    """
    Synchronous wrapper for async engine
    Maintains compatibility with existing CLI code
    """

    def __init__(self, config: Optional[AIEngineConfig] = None):
        self.engine = DSMILAIEngineV2(config)

    def generate(self, prompt: str, model_selection: str = "fast") -> dict:
        """
        Synchronous generate method (legacy compatibility)

        Returns dict for backward compatibility with existing code
        """
        try:
            model_tier = ModelTier(model_selection)
        except ValueError:
            model_tier = ModelTier.FAST

        request = DSMILQueryRequest(prompt=prompt, model=model_tier)

        # Run async in event loop
        result = asyncio.run(self.engine.generate(request))

        # Convert to dict for legacy compatibility
        return {
            "success": True,
            "response": result.response,
            "model": result.model_used,
            "latency_ms": result.latency_ms,
        }


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Demonstrate Pydantic AI engine capabilities"""
    engine = DSMILAIEngineV2()

    # Example 1: Simple query with validation
    print("=" * 70)
    print("Example 1: Simple Query with Type Safety")
    print("=" * 70)

    request = DSMILQueryRequest(
        prompt="Explain how TPM attestation works",
        model=ModelTier.FAST,
        temperature=0.7
    )

    result = await engine.generate(request)
    print(f"Response: {result.response[:200]}...")
    print(f"Model: {result.model_used}")
    print(f"Latency: {result.latency_ms:.2f}ms")
    print()

    # Example 2: Structured code generation
    print("=" * 70)
    print("Example 2: Structured Code Generation")
    print("=" * 70)

    code_result = await engine.generate_code(
        "Create a function to hash a password securely",
        language="python",
        model=ModelTier.CODE
    )

    print(f"Code:\n{code_result.code}\n")
    print(f"Explanation: {code_result.explanation}\n")
    print(f"Security Notes: {code_result.security_notes}\n")
    print(f"Dependencies: {code_result.dependencies}\n")

    # Example 3: Security analysis
    print("=" * 70)
    print("Example 3: Security Analysis")
    print("=" * 70)

    dangerous_code = """
import os
def run_command(cmd):
    os.system(cmd)  # Dangerous!
    """

    security_analysis = await engine.analyze_security(dangerous_code, "python")
    print(f"Vulnerability Level: {security_analysis.vulnerability_level}")
    print(f"Type: {security_analysis.vulnerability_type}")
    print(f"Description: {security_analysis.description}")
    print(f"Remediation: {security_analysis.remediation}")


if __name__ == "__main__":
    if PYDANTIC_AI_AVAILABLE:
        asyncio.run(example_usage())
    else:
        print("Install Pydantic AI to run examples: pip install pydantic-ai")
