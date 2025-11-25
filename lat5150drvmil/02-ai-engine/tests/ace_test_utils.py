#!/usr/bin/env python3
"""
ACE-FCA Test Utilities Module
------------------------------
Provides mock classes and utilities for testing ACE-FCA components.

This module contains test doubles (mocks, stubs, fakes) that were previously
scattered across production modules. Separating them improves code organization
and makes it clear what is production code vs test code.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


# =============================================================================
# Mock AI Engine
# =============================================================================

class MockAI:
    """
    Simple mock AI for basic testing.

    Returns a generic mock response for any prompt.
    """

    def generate(self, prompt: str, model: str = "code", stream: bool = False):
        """Generate mock response"""
        return {"text": "Mock AI response"}


class MockAIEngine:
    """
    Mock AI Engine for ACE workflow testing.

    Simulates realistic AI responses for different workflow phases
    (Research, Planning, Implementation, Verification) to enable
    automated testing without actual AI calls.
    """

    def __init__(self, custom_responses: Optional[Dict[str, str]] = None):
        """
        Initialize mock AI engine.

        Args:
            custom_responses: Optional dict mapping keywords to custom responses
        """
        self.custom_responses = custom_responses or {}
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        model: str = "code",
        stream: bool = False
    ) -> Dict[str, str]:
        """
        Generate mock response based on prompt content.

        Args:
            prompt: The prompt text
            model: Model preference (ignored in mock)
            stream: Whether to stream (ignored in mock)

        Returns:
            Dict with 'text' key containing the response
        """
        # Record call
        self.call_history.append({
            'prompt': prompt,
            'model': model,
            'stream': stream
        })

        # Check for custom responses first
        for keyword, response in self.custom_responses.items():
            if keyword.upper() in prompt.upper():
                return {"text": response}

        # Default phase-based responses
        if "RESEARCH" in prompt:
            return {"text": """
## Research Findings:
- Architecture: REST API with Flask
- Key files: src/api/rate_limiter.py (doesn't exist yet), src/api/middleware.py
- Current pattern: No rate limiting implemented
- Constraint: Must not break existing endpoints
"""}
        elif "PLANNING" in prompt or "PLAN" in prompt:
            return {"text": """
## Implementation Plan:

**Phase 1**: Create rate limiter module
- File: src/api/rate_limiter.py
- Test: Test basic rate limiting logic

**Phase 2**: Add middleware integration
- File: src/api/middleware.py
- Test: Test middleware application

**Phase 3**: Add configuration
- File: config/rate_limits.json
- Test: Test different rate limit configs
"""}
        elif "IMPLEMENTATION" in prompt or "IMPLEMENT" in prompt:
            return {"text": "Created rate_limiter.py with token bucket algorithm. Added middleware integration. All phases completed successfully."}
        elif "VERIFICATION" in prompt or "VERIFY" in prompt:
            return {"text": "✓ All tests pass. ✓ Rate limiting works correctly. ✓ Existing endpoints unaffected."}
        else:
            return {"text": "Mock response"}

    def get_call_count(self) -> int:
        """Get number of generate() calls made"""
        return len(self.call_history)

    def get_last_prompt(self) -> Optional[str]:
        """Get the last prompt sent to generate()"""
        if self.call_history:
            return self.call_history[-1]['prompt']
        return None

    def clear_history(self):
        """Clear call history"""
        self.call_history.clear()


# =============================================================================
# Mock Subagent Result
# =============================================================================

@dataclass
class MockSubagentResult:
    """Mock subagent result for testing"""
    agent_type: str
    compressed_output: str
    raw_output: str
    metadata: Dict[str, Any]
    success: bool = True

    @classmethod
    def create_success(
        cls,
        agent_type: str,
        output: str = "Mock output",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a successful mock result"""
        return cls(
            agent_type=agent_type,
            compressed_output=output[:100] if len(output) > 100 else output,
            raw_output=output,
            metadata=metadata or {},
            success=True
        )

    @classmethod
    def create_failure(
        cls,
        agent_type: str,
        error_message: str = "Mock error"
    ):
        """Create a failed mock result"""
        return cls(
            agent_type=agent_type,
            compressed_output=f"Error: {error_message}",
            raw_output=f"Error: {error_message}",
            metadata={'error': error_message},
            success=False
        )


# =============================================================================
# Mock Workflow Task
# =============================================================================

@dataclass
class MockWorkflowTask:
    """Mock workflow task for testing"""
    description: str
    task_type: str = "feature"
    estimated_complexity: str = "medium"
    constraints: Optional[List[str]] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_compressed_output(output: str, max_tokens: int = 700):
    """
    Assert that output is properly compressed.

    Args:
        output: The compressed output
        max_tokens: Maximum allowed tokens (1 token ≈ 4 chars)

    Raises:
        AssertionError: If output exceeds token limit
    """
    max_chars = max_tokens * 4
    if len(output) > max_chars:
        raise AssertionError(
            f"Output too long: {len(output)} chars "
            f"(~{len(output)//4} tokens) exceeds {max_tokens} token limit"
        )


def assert_context_utilization(tokens_used: int, max_tokens: int, target_min: float = 0.4, target_max: float = 0.6):
    """
    Assert that context utilization is within target range.

    Args:
        tokens_used: Number of tokens used
        max_tokens: Maximum available tokens
        target_min: Minimum target utilization (0.0-1.0)
        target_max: Maximum target utilization (0.0-1.0)

    Raises:
        AssertionError: If utilization is outside target range
    """
    utilization = tokens_used / max_tokens
    if utilization < target_min:
        raise AssertionError(
            f"Context underutilized: {utilization:.1%} < {target_min:.1%}"
        )
    if utilization > target_max:
        raise AssertionError(
            f"Context overutilized: {utilization:.1%} > {target_max:.1%}"
        )


def assert_phase_output_format(output: str, phase_name: str):
    """
    Assert that phase output follows expected format.

    Args:
        output: The phase output
        phase_name: Name of the phase

    Raises:
        AssertionError: If output format is invalid
    """
    if not output or not isinstance(output, str):
        raise AssertionError(f"{phase_name} output must be non-empty string")

    if len(output) < 10:
        raise AssertionError(f"{phase_name} output too short: {len(output)} chars")


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    print("ACE-FCA Test Utilities Demo")
    print("=" * 80)

    # Demo MockAIEngine
    print("\n1. MockAIEngine Demo:")
    print("-" * 80)

    ai = MockAIEngine()

    # Test different phases
    phases = ["RESEARCH", "PLANNING", "IMPLEMENTATION", "VERIFICATION"]
    for phase in phases:
        response = ai.generate(f"Execute {phase} phase for task X")
        print(f"\n{phase} Response:")
        print(response['text'][:100] + "...")

    print(f"\nTotal calls: {ai.get_call_count()}")
    print(f"Last prompt: {ai.get_last_prompt()[:50]}...")

    # Demo custom responses
    print("\n\n2. Custom Responses Demo:")
    print("-" * 80)

    custom_ai = MockAIEngine(custom_responses={
        "authentication": "Custom auth response",
        "database": "Custom DB response"
    })

    response = custom_ai.generate("Implement authentication system")
    print(f"Custom response: {response['text']}")

    # Demo MockSubagentResult
    print("\n\n3. MockSubagentResult Demo:")
    print("-" * 80)

    success = MockSubagentResult.create_success("research", "Found 42 files with authentication code")
    print(f"Success result: {success.compressed_output}")
    print(f"Success: {success.success}")

    failure = MockSubagentResult.create_failure("implementation", "File not found")
    print(f"\nFailure result: {failure.compressed_output}")
    print(f"Success: {failure.success}")

    # Demo assertions
    print("\n\n4. Assertion Helpers Demo:")
    print("-" * 80)

    try:
        assert_compressed_output("Short output", max_tokens=700)
        print("✓ Compression assertion passed")
    except AssertionError as e:
        print(f"✗ Compression assertion failed: {e}")

    try:
        assert_context_utilization(tokens_used=4000, max_tokens=8192)
        print("✓ Context utilization assertion passed")
    except AssertionError as e:
        print(f"✗ Context utilization assertion failed: {e}")

    try:
        assert_phase_output_format("Valid output text", "Research")
        print("✓ Phase format assertion passed")
    except AssertionError as e:
        print(f"✗ Phase format assertion failed: {e}")

    print("\n" + "=" * 80)
    print("All demos completed!")
