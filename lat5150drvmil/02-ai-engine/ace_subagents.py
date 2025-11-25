#!/usr/bin/env python3
"""
ACE-FCA Specialized Subagents

Context-isolated subagents for specialized tasks:
- ResearchAgent: Codebase exploration and analysis
- PlannerAgent: Implementation planning
- ImplementerAgent: Code generation and modification
- VerifierAgent: Testing and validation
- SummarizerAgent: Content compression and summarization

Key principle: Subagents work with fresh context windows and return
COMPRESSED findings to parent agents. This prevents context pollution
and maintains optimal 40-60% utilization.

Based on ACE-FCA methodology:
https://github.com/humanlayer/advanced-context-engineering-for-coding-agents
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from ace_context_engine import ACEContextEngine, PhaseType
from ace_interfaces import (
    FileSystemInterface,
    CommandExecutorInterface,
    ReviewInterface,
    StandardFileSystem,
    SubprocessCommandExecutor,
    InteractiveReview
)
from ace_exceptions import (
    SubagentExecutionError,
    FileSystemError,
    CommandExecutionError,
    wrap_exception
)
from utilities.file_search import FileSearchUtility
from utilities.code_analyzer import CodeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SubagentResult:
    """Result from specialized subagent execution"""
    agent_type: str
    compressed_output: str  # Compressed findings (< 500 tokens)
    raw_output: str        # Full output (for logging)
    metadata: Dict
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "agent_type": self.agent_type,
            "compressed_output": self.compressed_output,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error
        }


class BaseSubagent(ABC):
    """Base class for specialized subagents"""

    def __init__(
        self,
        ai_engine,
        max_tokens: int = 4096,
        filesystem: Optional[FileSystemInterface] = None,
        command_executor: Optional[CommandExecutorInterface] = None,
        review_interface: Optional[ReviewInterface] = None
    ):
        """
        Initialize subagent with dedicated context window and dependencies

        Args:
            ai_engine: AI engine for inference
            max_tokens: Smaller context window for focused tasks
            filesystem: File system interface (creates default if None)
            command_executor: Command executor interface (creates default if None)
            review_interface: Review interface (creates default if None)
        """
        self.ai_engine = ai_engine
        self.ace = ACEContextEngine(max_tokens=max_tokens)
        self.agent_type = self.__class__.__name__

        # Dependency injection with defaults
        self.filesystem = filesystem or StandardFileSystem()
        self.command_executor = command_executor or SubprocessCommandExecutor()
        self.review_interface = review_interface or InteractiveReview(auto_approve=True)

    @abstractmethod
    def execute(self, task: Dict) -> SubagentResult:
        """Execute subagent task and return compressed result"""
        pass

    def _compress_output(self, raw_output: str, max_tokens: int = 500) -> str:
        """
        Compress output to max_tokens using AI summarization

        This is the KEY to ACE-FCA: return compressed findings only
        """
        if self.ace.estimate_tokens(raw_output) <= max_tokens:
            return raw_output

        # Use AI to compress
        compress_prompt = f"""
Compress the following output to under {max_tokens} tokens while preserving all critical information.
Focus on key findings, remove redundancy, use bullet points.

Output to compress:
{raw_output}

Compressed output:"""

        result = self.ai_engine.generate(
            compress_prompt,
            model="fast",  # Use fast model for compression
            stream=False
        )

        compressed = result.get("text", raw_output[:max_tokens * 4])  # Fallback: truncate

        return compressed


class ResearchAgent(BaseSubagent):
    """
    Specialized agent for codebase research and analysis

    Responsibilities:
    - File search and discovery (via FileSearchUtility)
    - Architecture analysis (via CodeAnalyzer)
    - Pattern identification (via CodeAnalyzer)
    - Dependency mapping

    Returns: Compressed architecture overview and relevant files

    Note: This agent now acts as an orchestrator, delegating specialized
    tasks to utility classes rather than performing low-level operations itself.
    """

    def __init__(
        self,
        ai_engine,
        max_tokens: int = 4096,
        filesystem: Optional[FileSystemInterface] = None,
        command_executor: Optional[CommandExecutorInterface] = None,
        review_interface: Optional[ReviewInterface] = None,
        file_search: Optional[FileSearchUtility] = None,
        code_analyzer: Optional[CodeAnalyzer] = None
    ):
        """
        Initialize ResearchAgent with utilities via dependency injection.

        Args:
            ai_engine: AI engine for inference
            max_tokens: Context window size
            filesystem: File system interface
            command_executor: Command executor interface
            review_interface: Review interface
            file_search: File search utility (creates default if None)
            code_analyzer: Code analyzer utility (creates default if None)
        """
        super().__init__(ai_engine, max_tokens, filesystem, command_executor, review_interface)

        # Initialize utilities (create defaults if not provided)
        self.file_search = file_search or FileSearchUtility(
            filesystem=self.filesystem,
            command_executor=self.command_executor
        )
        self.code_analyzer = code_analyzer or CodeAnalyzer(
            filesystem=self.filesystem
        )

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute research task

        Args:
            task: Dict with:
                - query: Research query
                - search_paths: Optional list of paths to search
                - file_patterns: Optional file patterns to match

        Returns:
            SubagentResult with compressed findings
        """
        query = task.get("query", "")
        search_paths = task.get("search_paths", ["."])
        file_patterns = task.get("file_patterns", ["*.py", "*.js", "*.ts"])

        raw_findings = []

        # 1. Search for relevant files using FileSearchUtility
        relevant_files = self.file_search.search_files(
            paths=search_paths,
            patterns=file_patterns,
            query=query,
            max_results=50
        )
        raw_findings.append(f"## Relevant Files ({len(relevant_files)} found):\n")
        raw_findings.append("\n".join(f"- {f}" for f in relevant_files[:20]))  # Top 20

        # 2. Analyze architecture using CodeAnalyzer
        architecture = self.code_analyzer.analyze_architecture(
            files=relevant_files,
            max_dirs=10
        )
        raw_findings.append(f"\n## Architecture:\n{architecture}")

        # 3. Find implementation patterns using CodeAnalyzer
        patterns = self.code_analyzer.find_patterns(
            files=relevant_files,
            query=query,
            max_files_to_sample=5,
            max_patterns=10
        )
        raw_findings.append(f"\n## Implementation Patterns:\n{patterns}")

        raw_output = "\n".join(raw_findings)

        # Compress findings
        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="ResearchAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "files_found": len(relevant_files),
                "search_paths": search_paths
            }
        )



class PlannerAgent(BaseSubagent):
    """
    Specialized agent for implementation planning

    Responsibilities:
    - Break down tasks into phases
    - Identify files to modify
    - Create testing strategy
    - Consider alternatives

    Returns: Compressed phased implementation plan
    """

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute planning task

        Args:
            task: Dict with:
                - description: Task description
                - research_findings: Output from ResearchAgent
                - constraints: List of constraints

        Returns:
            SubagentResult with compressed plan
        """
        description = task.get("description", "")
        research = task.get("research_findings", "")
        constraints = task.get("constraints", [])

        # Build planning prompt
        planning_prompt = f"""
You are a specialized PLANNING agent. Create a detailed implementation plan.

Task: {description}

Research Findings:
{research}

Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Create a plan with:
1. Implementation phases (3-5 phases)
2. Specific files to modify in each phase
3. Testing strategy for each phase
4. Risk assessment

Keep response under 600 tokens. Use structured format.
"""

        # Execute with AI
        result = self.ai_engine.generate(
            planning_prompt,
            model="quality_code",
            stream=False
        )

        raw_output = result.get("text", "")

        # Compress if needed
        compressed = self._compress_output(raw_output, max_tokens=600)

        return SubagentResult(
            agent_type="PlannerAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "task": description,
                "constraints_count": len(constraints)
            }
        )


class ImplementerAgent(BaseSubagent):
    """
    Specialized agent for code implementation

    Responsibilities:
    - Generate code based on plan
    - Make file modifications
    - Track changes made
    - Report issues

    Returns: Compressed implementation notes
    """

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute implementation task

        Args:
            task: Dict with:
                - plan: Implementation plan
                - phase: Current phase to implement
                - files: Files to modify

        Returns:
            SubagentResult with compressed implementation notes
        """
        plan = task.get("plan", "")
        phase = task.get("phase", "Phase 1")
        files = task.get("files", [])

        implementation_prompt = f"""
You are a specialized IMPLEMENTATION agent. Execute this phase:

Phase: {phase}

Plan Context:
{plan}

Files to modify: {', '.join(files)}

Generate the code changes needed. Be concise but complete.
Keep response under 500 tokens - let the code speak for itself.
"""

        # Execute with AI
        result = self.ai_engine.generate(
            implementation_prompt,
            model="uncensored_code",  # Use uncensored model
            stream=False
        )

        raw_output = result.get("text", "")

        # Compress if needed
        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="ImplementerAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "phase": phase,
                "files_modified": len(files)
            }
        )


class VerifierAgent(BaseSubagent):
    """
    Specialized agent for testing and verification

    Responsibilities:
    - Run tests
    - Check code quality
    - Validate functionality
    - Report issues

    Returns: Compressed verification results
    """

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute verification task

        Args:
            task: Dict with:
                - implementation_notes: Notes from implementation
                - test_command: Optional test command to run
                - files_changed: List of files that changed

        Returns:
            SubagentResult with compressed verification results
        """
        implementation = task.get("implementation_notes", "")
        test_command = task.get("test_command", "pytest")
        files_changed = task.get("files_changed", [])

        results = []

        # 1. Run tests if command provided
        if test_command:
            test_result = self._run_tests(test_command)
            results.append(f"## Test Results:\n{test_result}")

        # 2. Check syntax of changed files
        syntax_results = self._check_syntax(files_changed)
        results.append(f"\n## Syntax Check:\n{syntax_results}")

        # 3. AI-based verification
        verification_prompt = f"""
You are a specialized VERIFICATION agent. Review this implementation:

Implementation:
{implementation}

Files changed: {', '.join(files_changed)}

Verify:
1. Functionality: Does it work as intended?
2. Code quality: Any issues?
3. Edge cases: What could go wrong?

Keep response under 400 tokens. Focus on pass/fail status.
"""

        ai_verification = self.ai_engine.generate(
            verification_prompt,
            model="quality_code",
            stream=False
        )

        results.append(f"\n## AI Verification:\n{ai_verification.get('text', '')}")

        raw_output = "\n".join(results)

        # Compress if needed
        compressed = self._compress_output(raw_output, max_tokens=500)

        return SubagentResult(
            agent_type="VerifierAgent",
            compressed_output=compressed,
            raw_output=raw_output,
            metadata={
                "test_command": test_command,
                "files_checked": len(files_changed)
            }
        )

    def _run_tests(self, command: str) -> str:
        """Run test command and return results"""
        try:
            # Use command executor interface
            result = self.command_executor.execute(command, timeout=60)

            if result.success:
                return "✓ All tests passed"
            else:
                # Return last 500 chars of error output
                error = result.stderr or result.stdout
                return f"✗ Tests failed:\n{error[-500:]}"

        except Exception as e:
            if "timeout" in str(e).lower():
                return "✗ Tests timed out (>60s)"
            return f"✗ Could not run tests: {str(e)}"

    def _check_syntax(self, files: List[str]) -> str:
        """Check syntax of files"""
        results = []

        for file in files:
            if not self.filesystem.file_exists(file):
                continue

            if file.endswith('.py'):
                try:
                    # Use command executor interface
                    result = self.command_executor.execute(
                        f"python -m py_compile {file}",
                        timeout=5
                    )
                    if result.success:
                        results.append(f"✓ {file}")
                    else:
                        results.append(f"✗ {file}: {result.stderr[:100]}")
                except Exception as e:
                    results.append(f"✗ {file}: {str(e)}")

        return "\n".join(results) if results else "No files to check"


class SummarizerAgent(BaseSubagent):
    """
    Specialized agent for content compression and summarization

    Responsibilities:
    - Compress large outputs
    - Extract key information
    - Remove redundancy
    - Maintain critical details

    Returns: Ultra-compressed summary
    """

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute summarization task

        Args:
            task: Dict with:
                - content: Content to summarize
                - max_tokens: Target token count
                - focus: What to focus on (optional)

        Returns:
            SubagentResult with compressed content
        """
        content = task.get("content", "")
        max_tokens = task.get("max_tokens", 300)
        focus = task.get("focus", "key findings")

        summarize_prompt = f"""
Summarize the following content to under {max_tokens} tokens.
Focus on: {focus}

Content:
{content}

Ultra-compressed summary:"""

        result = self.ai_engine.generate(
            summarize_prompt,
            model="fast",
            stream=False
        )

        compressed = result.get("text", "")

        return SubagentResult(
            agent_type="SummarizerAgent",
            compressed_output=compressed,
            raw_output=content,
            metadata={
                "original_tokens": self.ace.estimate_tokens(content),
                "compressed_tokens": self.ace.estimate_tokens(compressed),
                "compression_ratio": round(
                    self.ace.estimate_tokens(compressed) / max(self.ace.estimate_tokens(content), 1),
                    2
                )
            }
        )


# =============================================================================
# Subagent Registration (using registry pattern)
# =============================================================================

from ace_registry import (
    register_subagent,
    create_subagent as registry_create_subagent,
    SubagentCapability
)

# Register all subagents
register_subagent(
    agent_type="research",
    agent_class=ResearchAgent,
    capabilities=[SubagentCapability.RESEARCH, SubagentCapability.CODE_GENERATION],
    description="Codebase exploration and analysis",
    priority=10
)

register_subagent(
    agent_type="planner",
    agent_class=PlannerAgent,
    capabilities=[SubagentCapability.PLANNING],
    description="Implementation planning and strategy",
    priority=10
)

register_subagent(
    agent_type="implementer",
    agent_class=ImplementerAgent,
    capabilities=[SubagentCapability.IMPLEMENTATION, SubagentCapability.CODE_GENERATION],
    description="Code generation and modification",
    priority=10
)

register_subagent(
    agent_type="verifier",
    agent_class=VerifierAgent,
    capabilities=[SubagentCapability.VERIFICATION, SubagentCapability.TESTING],
    description="Testing and validation",
    priority=10
)

register_subagent(
    agent_type="summarizer",
    agent_class=SummarizerAgent,
    capabilities=[SubagentCapability.DOCUMENTATION],
    description="Content compression and summarization",
    priority=5
)


def create_subagent(agent_type: str, ai_engine, **kwargs) -> BaseSubagent:
    """
    Create a specialized subagent using the registry.

    Args:
        agent_type: Type of agent ('research', 'planner', 'implementer', 'verifier', 'summarizer')
        ai_engine: AI engine instance
        **kwargs: Additional arguments (filesystem, command_executor, review_interface, etc.)

    Returns:
        Specialized subagent instance
    """
    return registry_create_subagent(agent_type, ai_engine, **kwargs)


# Example usage
if __name__ == "__main__":
    from tests.ace_test_utils import MockAI

    print("ACE-FCA Specialized Subagents")
    print("=" * 60)

    # Create mock AI engine
    ai = MockAI()

    # Test ResearchAgent
    print("\n1. Testing ResearchAgent...")
    research_agent = create_subagent("research", ai)
    result = research_agent.execute({
        "query": "authentication",
        "search_paths": ["."],
        "file_patterns": ["*.py"]
    })
    print(f"   ✓ Found {result.metadata['files_found']} files")
    print(f"   ✓ Compressed output: {len(result.compressed_output)} chars")

    # Test PlannerAgent
    print("\n2. Testing PlannerAgent...")
    planner_agent = create_subagent("planner", ai)
    result = planner_agent.execute({
        "description": "Add rate limiting",
        "research_findings": "Files: api.py, middleware.py",
        "constraints": ["No breaking changes"]
    })
    print(f"   ✓ Plan generated")

    # Test SummarizerAgent
    print("\n3. Testing SummarizerAgent...")
    summarizer = create_subagent("summarizer", ai)
    result = summarizer.execute({
        "content": "Very long content " * 100,
        "max_tokens": 100
    })
    print(f"   ✓ Compression ratio: {result.metadata['compression_ratio']}")
