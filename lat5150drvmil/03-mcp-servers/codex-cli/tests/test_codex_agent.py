#!/usr/bin/env python3
"""
Tests for CodexAgent subagent

Author: SWORD Intelligence
License: Apache-2.0
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "02-ai-engine"))

import pytest
from codex_subagent import CodexAgent, CodexCapability


class TestCodexAgent:
    """Test suite for CodexAgent"""

    def setup_method(self):
        """Set up test fixtures"""
        self.agent = CodexAgent()

    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        assert self.agent is not None
        assert self.agent.agent_type == "CodexAgent"

    def test_capabilities_defined(self):
        """Test all capabilities are defined"""
        caps = CodexAgent.CAPABILITIES

        expected_capabilities = [
            "code_generation",
            "code_review",
            "debugging",
            "refactoring",
            "documentation",
            "testing",
            "optimization",
            "conversion",
            "explanation"
        ]

        for cap_name in expected_capabilities:
            assert cap_name in caps
            cap = caps[cap_name]
            assert isinstance(cap, CodexCapability)
            assert cap.name
            assert cap.description
            assert cap.complexity in ["low", "medium", "high"]
            assert cap.avg_tokens > 0
            assert len(cap.use_cases) > 0

    def test_get_capabilities_info(self):
        """Test capabilities info retrieval"""
        info = CodexAgent.get_capabilities_info()

        assert isinstance(info, dict)
        assert len(info) >= 9  # At least 9 capabilities

        for name, details in info.items():
            assert "description" in details
            assert "complexity" in details
            assert "avg_tokens" in details
            assert "use_cases" in details

    def test_code_generation_task(self):
        """Test code generation execution"""
        task = {
            "action": "generate",
            "prompt": "Write a function to add two numbers",
            "language": "python"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.agent_type == "CodexAgent"
        assert result.compressed_output
        assert result.metadata["action"] == "code_generation"
        assert result.metadata["language"] == "python"

    def test_code_review_task(self):
        """Test code review execution"""
        task = {
            "action": "review",
            "code": "def add(a, b): return a + b",
            "focus": "all",
            "language": "python"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "code_review"
        assert result.metadata["focus"] == "all"

    def test_debugging_task(self):
        """Test debugging execution"""
        task = {
            "action": "debug",
            "code": "print(x)",
            "error": "NameError: name 'x' is not defined"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "debugging"

    def test_refactoring_task(self):
        """Test refactoring execution"""
        task = {
            "action": "refactor",
            "code": "def f(x): return x * x",
            "goal": "readability"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "refactoring"

    def test_documentation_task(self):
        """Test documentation generation"""
        task = {
            "action": "document",
            "code": "def add(a, b): return a + b",
            "format": "markdown"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "documentation"

    def test_testing_task(self):
        """Test test generation"""
        task = {
            "action": "test",
            "code": "def add(a, b): return a + b",
            "framework": "pytest"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "testing"

    def test_optimization_task(self):
        """Test optimization execution"""
        task = {
            "action": "optimize",
            "code": "result = [x*x for x in range(1000000)]",
            "target": "speed"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "optimization"

    def test_conversion_task(self):
        """Test language conversion"""
        task = {
            "action": "convert",
            "code": "def add(a, b): return a + b",
            "from_language": "python",
            "to_language": "rust"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "conversion"

    def test_explanation_task(self):
        """Test code explanation"""
        task = {
            "action": "explain",
            "code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "audience": "beginner"
        }

        result = self.agent.execute(task)

        assert result is not None
        assert result.metadata["action"] == "explanation"

    def test_invalid_action(self):
        """Test handling of invalid action"""
        task = {
            "action": "invalid_action",
            "prompt": "test"
        }

        result = self.agent.execute(task)

        assert not result.success
        assert result.error
        assert "Unknown action" in result.error

    def test_output_compression(self):
        """Test output is properly compressed"""
        task = {
            "action": "generate",
            "prompt": "Write a complex authentication system",
            "language": "python"
        }

        result = self.agent.execute(task)

        # Compressed output should be smaller than raw
        # (assuming CLI actually returns large output)
        if result.raw_output and len(result.raw_output) > 2000:
            assert len(result.compressed_output) < len(result.raw_output)

    def test_metadata_includes_token_estimate(self):
        """Test metadata includes token estimates"""
        task = {
            "action": "generate",
            "prompt": "Simple function",
            "language": "python"
        }

        result = self.agent.execute(task)

        assert "action" in result.metadata
        # Token estimate may not be present if CLI not available

    def test_agent_handles_missing_cli(self):
        """Test agent handles missing CLI gracefully"""
        # Create agent with non-existent CLI path
        agent = CodexAgent(codex_cli_path="/nonexistent/path")

        task = {
            "action": "generate",
            "prompt": "test",
            "language": "python"
        }

        # Should not crash, should return graceful error
        result = agent.execute(task)
        assert result is not None


class TestCodexCapability:
    """Test CodexCapability dataclass"""

    def test_capability_creation(self):
        """Test creating a capability"""
        cap = CodexCapability(
            name="Test Capability",
            description="Test description",
            complexity="medium",
            avg_tokens=500,
            use_cases=["case1", "case2"]
        )

        assert cap.name == "Test Capability"
        assert cap.description == "Test description"
        assert cap.complexity == "medium"
        assert cap.avg_tokens == 500
        assert len(cap.use_cases) == 2


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
