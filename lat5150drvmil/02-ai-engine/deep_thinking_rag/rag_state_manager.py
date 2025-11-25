#!/usr/bin/env python3
"""
RAG State Manager (LangGraph-Style)

Manages state throughout the Deep-Thinking RAG pipeline.
Inspired by LangGraph's state management with checkpoints.

State flows through: Plan → Retrieve → Refine → Reflect → Critique → Synthesize

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class PipelinePhase(Enum):
    """RAG pipeline phases"""
    PLAN = "plan"
    RETRIEVE = "retrieve"
    REFINE = "refine"
    REFLECT = "reflect"
    CRITIQUE = "critique"
    SYNTHESIZE = "synthesize"
    COMPLETE = "complete"


class PolicyDecision(Enum):
    """Policy agent decisions"""
    CONTINUE = "continue"           # Continue to next phase
    RETRIEVE_MORE = "retrieve_more" # Need more evidence
    REVISE_QUERY = "revise_query"   # Replan with different strategy
    SYNTHESIZE = "synthesize"       # Ready to generate answer
    FAIL = "fail"                   # Cannot complete task


@dataclass
class RAGState:
    """
    Central state object passed between RAG pipeline nodes

    Inspired by LangGraph's state management:
    - Immutable snapshots at each step
    - Full history for rollback
    - Metadata for debugging
    """
    # Query information
    query: str
    sub_queries: List[str] = field(default_factory=list)

    # Retrieved documents
    documents: List[Dict] = field(default_factory=list)

    # Refined/reranked documents
    refined_documents: List[Dict] = field(default_factory=list)

    # Reflection assessment
    reflection: Optional[Dict] = None

    # Critique decision
    critique_decision: Optional[str] = None
    critique_reasoning: Optional[str] = None

    # Final answer
    answer: Optional[str] = None

    # Pipeline control
    current_phase: str = PipelinePhase.PLAN.value
    iteration: int = 0
    max_iterations: int = 10

    # Search strategy
    retrieval_strategy: str = "hybrid"  # "vector", "keyword", "hybrid"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict] = field(default_factory=list)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert state to dictionary"""
        return asdict(self)

    def log_step(self, phase: PipelinePhase, action: str, details: Dict):
        """Log a pipeline step to trace"""
        self.trace.append({
            "phase": phase.value,
            "action": action,
            "details": details,
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now().isoformat()


class RAGStateManager:
    """
    Manage RAG state with checkpoints for rollback

    Usage:
        manager = RAGStateManager()
        state = manager.create_state(query="How to optimize SQL?")

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(state)

        # Update state
        state.documents = [...]

        # Rollback if needed
        state = manager.load_checkpoint(checkpoint_id)
    """

    def __init__(self):
        """Initialize state manager"""
        self.checkpoints: Dict[str, RAGState] = {}
        self.checkpoint_counter = 0

    def create_state(
        self,
        query: str,
        max_iterations: int = 10,
        retrieval_strategy: str = "hybrid",
        metadata: Optional[Dict] = None
    ) -> RAGState:
        """
        Create new RAG state

        Args:
            query: User query
            max_iterations: Maximum pipeline iterations
            retrieval_strategy: Default retrieval strategy
            metadata: Optional metadata

        Returns:
            RAGState object
        """
        state = RAGState(
            query=query,
            max_iterations=max_iterations,
            retrieval_strategy=retrieval_strategy,
            metadata=metadata or {}
        )

        # Save initial checkpoint
        self.save_checkpoint(state, checkpoint_name="initial")

        return state

    def save_checkpoint(self, state: RAGState, checkpoint_name: Optional[str] = None) -> str:
        """
        Save state checkpoint for rollback

        Args:
            state: Current state
            checkpoint_name: Optional checkpoint name

        Returns:
            Checkpoint ID
        """
        if checkpoint_name is None:
            self.checkpoint_counter += 1
            checkpoint_name = f"checkpoint_{self.checkpoint_counter}"

        # Deep copy state
        import copy
        self.checkpoints[checkpoint_name] = copy.deepcopy(state)

        return checkpoint_name

    def load_checkpoint(self, checkpoint_name: str) -> Optional[RAGState]:
        """
        Load state from checkpoint

        Args:
            checkpoint_name: Checkpoint ID

        Returns:
            RAGState or None if not found
        """
        return self.checkpoints.get(checkpoint_name)

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs"""
        return list(self.checkpoints.keys())

    def clear_checkpoints(self):
        """Clear all checkpoints"""
        self.checkpoints.clear()
        self.checkpoint_counter = 0

    def get_trace(self, state: RAGState) -> List[Dict]:
        """Get execution trace from state"""
        return state.trace

    def get_statistics(self, state: RAGState) -> Dict:
        """Get state statistics"""
        return {
            "query": state.query,
            "current_phase": state.current_phase,
            "iteration": state.iteration,
            "max_iterations": state.max_iterations,
            "num_documents": len(state.documents),
            "num_refined_documents": len(state.refined_documents),
            "num_sub_queries": len(state.sub_queries),
            "has_answer": state.answer is not None,
            "critique_decision": state.critique_decision,
            "trace_length": len(state.trace),
            "elapsed_time": (
                datetime.fromisoformat(state.updated_at) -
                datetime.fromisoformat(state.created_at)
            ).total_seconds()
        }


if __name__ == "__main__":
    # Demo usage
    print("="*70)
    print("RAG State Manager Demo")
    print("="*70)

    manager = RAGStateManager()

    # Create initial state
    state = manager.create_state(
        query="How do I optimize PostgreSQL queries?",
        max_iterations=5,
        metadata={"user_id": "demo"}
    )

    print(f"\nInitial state created:")
    print(f"  Query: {state.query}")
    print(f"  Phase: {state.current_phase}")
    print(f"  Max iterations: {state.max_iterations}")

    # Simulate pipeline execution
    state.log_step(
        PipelinePhase.PLAN,
        "decompose_query",
        {"sub_queries": ["indexing", "EXPLAIN ANALYZE", "connection pooling"]}
    )
    state.sub_queries = ["indexing", "EXPLAIN ANALYZE", "connection pooling"]
    state.current_phase = PipelinePhase.RETRIEVE.value

    # Save checkpoint after plan
    checkpoint_id = manager.save_checkpoint(state, "after_plan")
    print(f"\n✓ Saved checkpoint: {checkpoint_id}")

    # Simulate retrieval
    state.log_step(
        PipelinePhase.RETRIEVE,
        "semantic_search",
        {"strategy": "hybrid", "num_documents": 10}
    )
    state.documents = [{"text": f"Document {i}", "score": 0.9 - i*0.1} for i in range(10)]
    state.current_phase = PipelinePhase.REFINE.value

    # Save checkpoint after retrieve
    checkpoint_id = manager.save_checkpoint(state, "after_retrieve")
    print(f"✓ Saved checkpoint: {checkpoint_id}")

    # Get statistics
    stats = manager.get_statistics(state)
    print(f"\nState statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show trace
    print(f"\nExecution trace:")
    for step in state.trace:
        print(f"  [{step['phase']}] {step['action']}: {step['details']}")

    # Demonstrate rollback
    print(f"\n✓ Available checkpoints: {manager.list_checkpoints()}")
    rolled_back_state = manager.load_checkpoint("after_plan")
    print(f"✓ Rolled back to 'after_plan'")
    print(f"  Current phase: {rolled_back_state.current_phase}")
    print(f"  Num documents: {len(rolled_back_state.documents)}")
