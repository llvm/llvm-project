"""
Enhanced Memory System with LangGraph Checkpoints

Implements automatic state management and cross-session memory:
- LangGraph-style checkpoints for rollback
- PostgreSQL + pgvector for semantic memory
- Cross-session context retrieval

Based on "Building Long-Term Memory in Agentic AI"
"""

from .langgraph_checkpoint_manager import CheckpointManager, Checkpoint

__all__ = ['CheckpointManager', 'Checkpoint']
