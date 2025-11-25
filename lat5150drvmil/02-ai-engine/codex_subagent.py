#!/usr/bin/env python3
"""
CodexAgent - Specialized subagent for OpenAI Codex CLI integration

Provides comprehensive developer support through OpenAI's Codex models,
integrated with the LAT5150DRVMIL agent orchestration system.

Features:
- Code generation, review, debugging, refactoring
- Test generation and documentation
- Code optimization and conversion
- Streaming responses
- Extended reasoning capabilities

Based on ACE-FCA methodology for optimal context utilization.

Author: SWORD Intelligence
License: Apache-2.0
"""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ace_subagents import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)


@dataclass
class CodexCapability:
    """Capability definition for Codex agent"""
    name: str
    description: str
    complexity: str  # low, medium, high
    avg_tokens: int  # Average token usage
    use_cases: List[str]


class CodexAgent(BaseSubagent):
    """
    Specialized agent for OpenAI Codex CLI integration

    Provides access to GPT-5-Codex and GPT-5-Codex-Mini models for:
    - Advanced code generation
    - Intelligent code review
    - Debugging assistance
    - Code refactoring
    - Documentation generation
    - Test generation
    - Code optimization
    - Language conversion

    This agent integrates with the local Codex CLI client and provides
    compressed results following ACE-FCA principles.
    """

    # Define Codex capabilities
    CAPABILITIES = {
        "code_generation": CodexCapability(
            name="Code Generation",
            description="Generate production-ready code from natural language",
            complexity="medium",
            avg_tokens=800,
            use_cases=[
                "Implementing new features",
                "Creating utilities and helpers",
                "Prototyping algorithms",
                "Writing boilerplate code"
            ]
        ),
        "code_review": CodexCapability(
            name="Code Review",
            description="Review code for bugs, security, performance, and style",
            complexity="high",
            avg_tokens=1200,
            use_cases=[
                "Security audits",
                "Performance analysis",
                "Best practices validation",
                "Code quality assessment"
            ]
        ),
        "debugging": CodexCapability(
            name="Debugging",
            description="Diagnose and fix code issues",
            complexity="high",
            avg_tokens=1000,
            use_cases=[
                "Bug fixing",
                "Error analysis",
                "Root cause identification",
                "Solution implementation"
            ]
        ),
        "refactoring": CodexCapability(
            name="Refactoring",
            description="Improve code quality and maintainability",
            complexity="medium",
            avg_tokens=900,
            use_cases=[
                "Code cleanup",
                "Architecture improvements",
                "Performance optimization",
                "Technical debt reduction"
            ]
        ),
        "documentation": CodexCapability(
            name="Documentation",
            description="Generate comprehensive code documentation",
            complexity="low",
            avg_tokens=600,
            use_cases=[
                "API documentation",
                "Inline comments",
                "Usage guides",
                "Architecture docs"
            ]
        ),
        "testing": CodexCapability(
            name="Test Generation",
            description="Generate comprehensive unit tests",
            complexity="medium",
            avg_tokens=800,
            use_cases=[
                "Unit tests",
                "Integration tests",
                "Test coverage",
                "Edge case testing"
            ]
        ),
        "optimization": CodexCapability(
            name="Code Optimization",
            description="Optimize code for speed, memory, or efficiency",
            complexity="high",
            avg_tokens=1100,
            use_cases=[
                "Performance tuning",
                "Memory optimization",
                "Algorithm improvements",
                "Bottleneck removal"
            ]
        ),
        "conversion": CodexCapability(
            name="Language Conversion",
            description="Convert code between programming languages",
            complexity="high",
            avg_tokens=1000,
            use_cases=[
                "Migration projects",
                "Cross-platform development",
                "Language modernization",
                "Framework porting"
            ]
        ),
        "explanation": CodexCapability(
            name="Code Explanation",
            description="Explain code functionality in natural language",
            complexity="low",
            avg_tokens=500,
            use_cases=[
                "Code understanding",
                "Learning and training",
                "Onboarding",
                "Knowledge transfer"
            ]
        ),
    }

    def __init__(self, ai_engine=None, codex_cli_path: Optional[str] = None, max_tokens: int = 8192):
        """
        Initialize Codex agent

        Args:
            ai_engine: AI engine (not used, Codex has its own)
            codex_cli_path: Path to codex-cli binary
            max_tokens: Maximum tokens per request
        """
        super().__init__(ai_engine, max_tokens)

        # Find codex-cli binary
        if codex_cli_path:
            self.codex_cli_path = Path(codex_cli_path)
        else:
            # Default locations
            possible_paths = [
                Path(__file__).parent.parent / "03-mcp-servers" / "codex-cli" / "target" / "release" / "codex-cli",
                Path(__file__).parent.parent / "03-mcp-servers" / "codex-cli" / "target" / "debug" / "codex-cli",
                Path.home() / ".cargo" / "bin" / "codex-cli",
            ]

            for path in possible_paths:
                if path.exists():
                    self.codex_cli_path = path
                    break
            else:
                logger.warning("Codex CLI binary not found in default locations")
                self.codex_cli_path = None

        if self.codex_cli_path:
            logger.info(f"Codex agent initialized with CLI at: {self.codex_cli_path}")
        else:
            logger.warning("Codex agent initialized without CLI binary (limited functionality)")

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute Codex task

        Args:
            task: Dict with:
                - action: Type of action (generate, review, debug, refactor, etc.)
                - prompt: Main prompt or description
                - code: Code to process (for review, debug, refactor, etc.)
                - language: Programming language
                - Additional parameters based on action

        Returns:
            SubagentResult with compressed findings
        """
        action = task.get("action", "generate")

        if action not in self.CAPABILITIES:
            return SubagentResult(
                agent_type="CodexAgent",
                compressed_output=f"Unknown action: {action}",
                raw_output=f"Unknown action: {action}. Available actions: {', '.join(self.CAPABILITIES.keys())}",
                metadata={"error": "invalid_action"},
                success=False,
                error=f"Unknown action: {action}"
            )

        # Route to appropriate handler
        handlers = {
            "generate": self._handle_code_generation,
            "code_generation": self._handle_code_generation,
            "review": self._handle_code_review,
            "code_review": self._handle_code_review,
            "debug": self._handle_debugging,
            "debugging": self._handle_debugging,
            "refactor": self._handle_refactoring,
            "refactoring": self._handle_refactoring,
            "document": self._handle_documentation,
            "documentation": self._handle_documentation,
            "test": self._handle_testing,
            "testing": self._handle_testing,
            "optimize": self._handle_optimization,
            "optimization": self._handle_optimization,
            "convert": self._handle_conversion,
            "conversion": self._handle_conversion,
            "explain": self._handle_explanation,
            "explanation": self._handle_explanation,
        }

        handler = handlers.get(action)
        if not handler:
            return SubagentResult(
                agent_type="CodexAgent",
                compressed_output=f"No handler for action: {action}",
                raw_output="",
                metadata={"error": "no_handler"},
                success=False,
                error=f"No handler for action: {action}"
            )

        try:
            return handler(task)
        except Exception as e:
            logger.error(f"Codex agent error: {e}", exc_info=True)
            return SubagentResult(
                agent_type="CodexAgent",
                compressed_output=f"Error: {str(e)}",
                raw_output=f"Error executing {action}: {str(e)}",
                metadata={"error": str(e)},
                success=False,
                error=str(e)
            )

    def _execute_codex_cli(self, prompt: str) -> Tuple[str, bool]:
        """
        Execute codex CLI with prompt

        Returns:
            (output, success)
        """
        if not self.codex_cli_path or not self.codex_cli_path.exists():
            return ("Codex CLI not available - using simulated response", False)

        try:
            result = subprocess.run(
                [str(self.codex_cli_path), "exec", prompt],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                return (result.stdout.strip(), True)
            else:
                error_msg = result.stderr.strip()
                logger.error(f"Codex CLI error: {error_msg}")
                return (f"Error: {error_msg}", False)

        except subprocess.TimeoutExpired:
            logger.error("Codex CLI timeout")
            return ("Error: Request timeout (>2 minutes)", False)

        except Exception as e:
            logger.error(f"Codex CLI execution error: {e}")
            return (f"Error: {str(e)}", False)

    def _handle_code_generation(self, task: Dict) -> SubagentResult:
        """Handle code generation task"""
        description = task.get("prompt", task.get("description", ""))
        language = task.get("language", "python")
        style = task.get("style", "clean")

        prompt = f"""Generate {style} {language} code for:

{description}

Requirements:
- Production-ready code
- Proper error handling
- Clear comments where needed
- Follow best practices

Provide clean, working code."""

        raw_output, success = self._execute_codex_cli(prompt)

        # Compress output
        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "code_generation",
                "language": language,
                "style": style,
                "tokens_estimate": self.ace.estimate_tokens(raw_output)
            },
            success=success
        )

    def _handle_code_review(self, task: Dict) -> SubagentResult:
        """Handle code review task"""
        code = task.get("code", "")
        focus = task.get("focus", "all")
        language = task.get("language", "auto-detect")

        prompt = f"""Review this {language} code with focus on {focus}:

```
{code}
```

Analyze for:
1. Security vulnerabilities
2. Performance issues
3. Code quality and style
4. Best practices compliance

Provide specific, actionable feedback."""

        raw_output, success = self._execute_codex_cli(prompt)

        # Compress output
        compressed = self._compress_output(raw_output, max_tokens=600)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "code_review",
                "focus": focus,
                "language": language,
                "code_length": len(code)
            },
            success=success
        )

    def _handle_debugging(self, task: Dict) -> SubagentResult:
        """Handle debugging task"""
        code = task.get("code", "")
        error = task.get("error", "")
        context = task.get("context", "")

        prompt = f"""Debug this code issue:

Code:
```
{code}
```

Error: {error}
{f"Context: {context}" if context else ""}

Provide:
1. Root cause
2. Why it happens
3. Fixed code
4. Prevention tips"""

        raw_output, success = self._execute_codex_cli(prompt)

        # Compress output
        compressed = self._compress_output(raw_output, max_tokens=600)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "debugging",
                "error_type": error[:100] if error else "unknown"
            },
            success=success
        )

    def _handle_refactoring(self, task: Dict) -> SubagentResult:
        """Handle refactoring task"""
        code = task.get("code", "")
        goal = task.get("goal", "general improvement")

        prompt = f"""Refactor this code for {goal}:

```
{code}
```

Improve quality while maintaining functionality.
Explain changes made."""

        raw_output, success = self._execute_codex_cli(prompt)

        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "refactoring",
                "goal": goal
            },
            success=success
        )

    def _handle_documentation(self, task: Dict) -> SubagentResult:
        """Handle documentation generation"""
        code = task.get("code", "")
        format_type = task.get("format", "markdown")

        prompt = f"""Generate {format_type} documentation for:

```
{code}
```

Include:
- Purpose and overview
- Parameters and returns
- Usage examples
- Edge cases"""

        raw_output, success = self._execute_codex_cli(prompt)

        compressed = self._compress_output(raw_output, max_tokens=400)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "documentation",
                "format": format_type
            },
            success=success
        )

    def _handle_testing(self, task: Dict) -> SubagentResult:
        """Handle test generation"""
        code = task.get("code", "")
        framework = task.get("framework", "pytest")

        prompt = f"""Generate {framework} tests for:

```
{code}
```

Include:
- Happy path tests
- Edge cases
- Error cases
- Good coverage"""

        raw_output, success = self._execute_codex_cli(prompt)

        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "testing",
                "framework": framework
            },
            success=success
        )

    def _handle_optimization(self, task: Dict) -> SubagentResult:
        """Handle code optimization"""
        code = task.get("code", "")
        target = task.get("target", "speed")

        prompt = f"""Optimize this code for {target}:

```
{code}
```

Provide optimized version with explanation."""

        raw_output, success = self._execute_codex_cli(prompt)

        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "optimization",
                "target": target
            },
            success=success
        )

    def _handle_conversion(self, task: Dict) -> SubagentResult:
        """Handle language conversion"""
        code = task.get("code", "")
        from_lang = task.get("from_language", "")
        to_lang = task.get("to_language", "")

        prompt = f"""Convert this {from_lang} code to {to_lang}:

```{from_lang}
{code}
```

Provide idiomatic {to_lang} code with notes."""

        raw_output, success = self._execute_codex_cli(prompt)

        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "conversion",
                "from_language": from_lang,
                "to_language": to_lang
            },
            success=success
        )

    def _handle_explanation(self, task: Dict) -> SubagentResult:
        """Handle code explanation"""
        code = task.get("code", "")
        audience = task.get("audience", "intermediate")

        prompt = f"""Explain this code for {audience} developers:

```
{code}
```

Provide clear, educational explanation."""

        raw_output, success = self._execute_codex_cli(prompt)

        compressed = self._compress_output(raw_output, max_tokens=400)

        return SubagentResult(
            agent_type="CodexAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "action": "explanation",
                "audience": audience
            },
            success=success
        )

    @classmethod
    def get_capabilities_info(cls) -> Dict:
        """Get detailed information about Codex capabilities"""
        return {
            name: {
                "description": cap.description,
                "complexity": cap.complexity,
                "avg_tokens": cap.avg_tokens,
                "use_cases": cap.use_cases
            }
            for name, cap in cls.CAPABILITIES.items()
        }


# Integration with agent orchestrator
def register_codex_agent(orchestrator):
    """
    Register Codex agent with orchestrator

    Args:
        orchestrator: Agent orchestrator instance
    """
    codex_agent = CodexAgent()

    # Add to orchestrator's agent registry
    # (Implementation depends on orchestrator API)

    logger.info("Codex agent registered with orchestrator")

    return codex_agent


if __name__ == "__main__":
    # Test Codex agent
    print("Codex Agent - Capability Test")
    print("=" * 70)

    agent = CodexAgent()

    # Test code generation
    print("\n1. Testing Code Generation...")
    result = agent.execute({
        "action": "generate",
        "prompt": "Write a Python function to calculate Fibonacci numbers",
        "language": "python"
    })
    print(f"   Success: {result.success}")
    print(f"   Output: {result.compressed_output[:200]}...")

    # Print capabilities
    print("\n2. Available Capabilities:")
    capabilities = CodexAgent.get_capabilities_info()
    for name, info in capabilities.items():
        print(f"\n   {name.upper()}:")
        print(f"   - {info['description']}")
        print(f"   - Complexity: {info['complexity']}")
        print(f"   - Avg tokens: {info['avg_tokens']}")
