#!/usr/bin/env python3
"""
ACE-FCA Phase Executor Module
------------------------------
Provides abstraction for workflow phase execution, separating phase logic
from orchestration to improve testability and follow Single Responsibility Principle.

Addresses: Multi-responsibility orchestrator, phase logic coupling, testability issues
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """Status of a phase execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class PhaseInput:
    """Input to a phase execution"""
    task_description: str
    context: Dict[str, Any]
    previous_phase_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PhaseOutput:
    """Output from a phase execution"""
    phase_name: str
    status: PhaseStatus
    output: str
    compressed_output: Optional[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    tokens_used: int = 0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PhaseExecutionContext:
    """Context for phase execution"""
    phase_name: str
    config: Dict[str, Any]
    subagent_type: str
    retry_count: int = 0
    max_retries: int = 3


# =============================================================================
# Abstract Phase Executor
# =============================================================================

class PhaseExecutor(ABC):
    """
    Abstract base class for phase executors.

    Each phase executor is responsible for executing a specific workflow phase
    (Research, Planning, Implementation, Verification) with its own logic
    and configuration.
    """

    def __init__(
        self,
        phase_name: str,
        subagent_type: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.phase_name = phase_name
        self.subagent_type = subagent_type
        self.config = config or {}
        self.execution_history: List[PhaseOutput] = []

    @abstractmethod
    def execute(self, phase_input: PhaseInput) -> PhaseOutput:
        """
        Execute the phase.

        Args:
            phase_input: Input to the phase

        Returns:
            PhaseOutput with results
        """
        pass

    def can_skip(self, phase_input: PhaseInput) -> bool:
        """
        Determine if this phase can be skipped.

        Override this method to implement phase-specific skip logic.

        Args:
            phase_input: Input to the phase

        Returns:
            True if phase can be skipped
        """
        return False

    def requires_review(self, phase_output: PhaseOutput) -> bool:
        """
        Determine if this phase output requires human review.

        Override this method to implement phase-specific review logic.

        Args:
            phase_output: Output from the phase

        Returns:
            True if review is required
        """
        return self.config.get('requires_review', False)

    def compress_output(
        self,
        output: str,
        target_tokens: Optional[int] = None
    ) -> str:
        """
        Compress phase output for context management.

        Override this method to implement phase-specific compression.

        Args:
            output: Raw output to compress
            target_tokens: Target token count (optional)

        Returns:
            Compressed output
        """
        # Default: truncate to target tokens (rough estimate: 1 token ≈ 4 chars)
        if target_tokens:
            max_chars = target_tokens * 4
            if len(output) > max_chars:
                return output[:max_chars] + "..."
        return output

    def get_execution_history(self) -> List[PhaseOutput]:
        """Get history of phase executions"""
        return self.execution_history.copy()

    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()


# =============================================================================
# Concrete Phase Executors
# =============================================================================

class SubagentPhaseExecutor(PhaseExecutor):
    """
    Phase executor that delegates to a subagent.

    This is the standard executor that uses the subagent pattern
    for phase execution.
    """

    def __init__(
        self,
        phase_name: str,
        subagent_type: str,
        subagent_factory,  # Callable that creates subagent
        config: Optional[Dict[str, Any]] = None,
        compressor=None  # Optional compressor function
    ):
        super().__init__(phase_name, subagent_type, config)
        self.subagent_factory = subagent_factory
        self.compressor = compressor

    def execute(self, phase_input: PhaseInput) -> PhaseOutput:
        """Execute phase using subagent"""
        logger.info(f"Executing phase: {self.phase_name} with subagent: {self.subagent_type}")

        try:
            # Create subagent
            subagent = self.subagent_factory(self.subagent_type)

            # Prepare task
            task = {
                'action': 'execute_phase',
                'phase': self.phase_name,
                'description': phase_input.task_description,
                'context': phase_input.context,
                'previous_output': phase_input.previous_phase_output,
                'config': self.config
            }

            # Execute
            result = subagent.execute(task)

            # Extract output
            output = result.raw_output if hasattr(result, 'raw_output') else str(result)
            compressed = result.compressed_output if hasattr(result, 'compressed_output') else None

            # Apply compression if not already compressed
            if compressed is None and self.compressor:
                target_tokens = self.config.get('max_tokens', 500)
                compressed = self.compressor(output, target_tokens)

            # Create phase output
            phase_output = PhaseOutput(
                phase_name=self.phase_name,
                status=PhaseStatus.COMPLETED,
                output=output,
                compressed_output=compressed,
                metadata=result.metadata if hasattr(result, 'metadata') else {},
                tokens_used=len(output) // 4  # Rough estimate
            )

            self.execution_history.append(phase_output)
            return phase_output

        except Exception as e:
            logger.error(f"Phase {self.phase_name} failed: {e}")
            phase_output = PhaseOutput(
                phase_name=self.phase_name,
                status=PhaseStatus.FAILED,
                output="",
                error=str(e)
            )
            self.execution_history.append(phase_output)
            return phase_output


class MockPhaseExecutor(PhaseExecutor):
    """
    Mock phase executor for testing.

    Returns pre-configured outputs without actually executing.
    """

    def __init__(
        self,
        phase_name: str,
        subagent_type: str,
        mock_output: str = "Mock output",
        mock_status: PhaseStatus = PhaseStatus.COMPLETED,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(phase_name, subagent_type, config)
        self.mock_output = mock_output
        self.mock_status = mock_status

    def execute(self, phase_input: PhaseInput) -> PhaseOutput:
        """Return mock output"""
        logger.info(f"Mock executing phase: {self.phase_name}")

        phase_output = PhaseOutput(
            phase_name=self.phase_name,
            status=self.mock_status,
            output=self.mock_output,
            compressed_output=self.mock_output[:100],  # Mock compression
            metadata={'mock': True}
        )

        self.execution_history.append(phase_output)
        return phase_output


# =============================================================================
# Phase Execution Pipeline
# =============================================================================

class PhaseExecutionPipeline:
    """
    Manages execution of multiple phases in sequence.

    Handles phase chaining, error recovery, and context passing.
    """

    def __init__(self, phases: List[PhaseExecutor]):
        self.phases = phases
        self.execution_log: List[PhaseOutput] = []

    def execute_all(
        self,
        initial_input: PhaseInput,
        stop_on_error: bool = True,
        enable_skip: bool = True
    ) -> List[PhaseOutput]:
        """
        Execute all phases in sequence.

        Args:
            initial_input: Input to the first phase
            stop_on_error: Stop pipeline if any phase fails
            enable_skip: Allow phases to be skipped

        Returns:
            List of PhaseOutput for all executed phases
        """
        results = []
        current_input = initial_input

        for phase in self.phases:
            # Check if phase can be skipped
            if enable_skip and phase.can_skip(current_input):
                logger.info(f"Skipping phase: {phase.phase_name}")
                output = PhaseOutput(
                    phase_name=phase.phase_name,
                    status=PhaseStatus.SKIPPED,
                    output="Phase skipped",
                    metadata={'skipped': True}
                )
                results.append(output)
                continue

            # Execute phase
            output = phase.execute(current_input)
            results.append(output)
            self.execution_log.append(output)

            # Check for failure
            if output.status == PhaseStatus.FAILED and stop_on_error:
                logger.error(f"Pipeline stopped due to failure in {phase.phase_name}")
                break

            # Prepare input for next phase
            current_input = PhaseInput(
                task_description=initial_input.task_description,
                context=initial_input.context,
                previous_phase_output=output.compressed_output or output.output,
                metadata=output.metadata
            )

        return results

    def execute_single(
        self,
        phase_index: int,
        phase_input: PhaseInput
    ) -> PhaseOutput:
        """
        Execute a single phase by index.

        Args:
            phase_index: Index of the phase to execute
            phase_input: Input to the phase

        Returns:
            PhaseOutput from the executed phase
        """
        if phase_index < 0 or phase_index >= len(self.phases):
            raise ValueError(f"Invalid phase index: {phase_index}")

        phase = self.phases[phase_index]
        output = phase.execute(phase_input)
        self.execution_log.append(output)
        return output

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        return {
            'total_phases': len(self.phases),
            'executed_phases': len(self.execution_log),
            'completed': sum(1 for o in self.execution_log if o.status == PhaseStatus.COMPLETED),
            'failed': sum(1 for o in self.execution_log if o.status == PhaseStatus.FAILED),
            'skipped': sum(1 for o in self.execution_log if o.status == PhaseStatus.SKIPPED),
            'total_tokens': sum(o.tokens_used for o in self.execution_log)
        }

    def clear_log(self):
        """Clear execution log"""
        self.execution_log.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_standard_pipeline(
    subagent_factory,
    compressor=None,
    config: Optional[Dict[str, Any]] = None
) -> PhaseExecutionPipeline:
    """
    Create a standard ACE-FCA workflow pipeline.

    Args:
        subagent_factory: Factory function for creating subagents
        compressor: Optional compression function
        config: Optional configuration

    Returns:
        PhaseExecutionPipeline with Research → Planning → Implementation → Verification
    """
    config = config or {}

    phases = [
        SubagentPhaseExecutor(
            phase_name="Research",
            subagent_type="research",
            subagent_factory=subagent_factory,
            compressor=compressor,
            config=config.get('research', {})
        ),
        SubagentPhaseExecutor(
            phase_name="Planning",
            subagent_type="planning",
            subagent_factory=subagent_factory,
            compressor=compressor,
            config=config.get('planning', {})
        ),
        SubagentPhaseExecutor(
            phase_name="Implementation",
            subagent_type="implementation",
            subagent_factory=subagent_factory,
            compressor=compressor,
            config=config.get('implementation', {})
        ),
        SubagentPhaseExecutor(
            phase_name="Verification",
            subagent_type="verification",
            subagent_factory=subagent_factory,
            compressor=compressor,
            config=config.get('verification', {})
        )
    ]

    return PhaseExecutionPipeline(phases)


def create_mock_pipeline() -> PhaseExecutionPipeline:
    """Create a mock pipeline for testing"""
    phases = [
        MockPhaseExecutor("Research", "research", "Mock research output"),
        MockPhaseExecutor("Planning", "planning", "Mock planning output"),
        MockPhaseExecutor("Implementation", "implementation", "Mock implementation output"),
        MockPhaseExecutor("Verification", "verification", "Mock verification output")
    ]

    return PhaseExecutionPipeline(phases)


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Create mock pipeline
    pipeline = create_mock_pipeline()

    # Execute all phases
    initial_input = PhaseInput(
        task_description="Test task",
        context={'test': True}
    )

    results = pipeline.execute_all(initial_input)

    print("Execution Results:")
    print("=" * 80)
    for result in results:
        print(f"\nPhase: {result.phase_name}")
        print(f"Status: {result.status.value}")
        print(f"Output: {result.output}")
        print(f"Tokens: {result.tokens_used}")

    print("\n" + "=" * 80)
    print("Execution Summary:")
    import json
    print(json.dumps(pipeline.get_execution_summary(), indent=2))
