#!/usr/bin/env python3
"""
LangGraph-Style Checkpoint Manager

Automatic state persistence with rollback support.

Based on "Building Long-Term Memory in Agentic AI" by Fareed Khan

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

import sqlite3
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Checkpoint:
    """State checkpoint"""
    checkpoint_id: str
    thread_id: str
    state: Dict[str, Any]
    parent_checkpoint_id: Optional[str]
    metadata: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


class CheckpointManager:
    """
    Automatic state checkpointing inspired by LangGraph

    Features:
    - Automatic state persistence
    - Rollback to previous states
    - Branch conversations ("what-if" scenarios)
    - Cross-session continuation

    Usage:
        manager = CheckpointManager()

        # Save state
        checkpoint_id = manager.save_checkpoint(
            thread_id="conversation_123",
            state={"messages": [...], "context": {...}}
        )

        # Load state
        state = manager.load_checkpoint(checkpoint_id)

        # Rollback
        previous_state = manager.rollback(thread_id, steps=2)
    """

    def __init__(self, db_path: str = "~/.rag_index/checkpoints.db"):
        """
        Initialize checkpoint manager

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()

    def _init_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                state TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)

        # Indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread ON checkpoints(thread_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON checkpoints(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent ON checkpoints(parent_checkpoint_id)")

        self.conn.commit()

    def save_checkpoint(
        self,
        thread_id: str,
        state: Dict[str, Any],
        parent_checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save state checkpoint

        Args:
            thread_id: Conversation thread ID
            state: Current state dict
            parent_checkpoint_id: Parent checkpoint (for branching)
            metadata: Optional metadata

        Returns:
            Checkpoint ID
        """
        import hashlib
        checkpoint_id = "ckpt_" + hashlib.sha256(
            f"{thread_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            state=state,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata or {},
            created_at=datetime.now().isoformat()
        )

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)
        """, (
            checkpoint.checkpoint_id,
            checkpoint.thread_id,
            json.dumps(checkpoint.state),
            checkpoint.parent_checkpoint_id,
            json.dumps(checkpoint.metadata),
            checkpoint.created_at
        ))

        self.conn.commit()

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load state from checkpoint

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            State dict or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT state FROM checkpoints WHERE checkpoint_id = ?
        """, (checkpoint_id,))

        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None

    def get_latest_checkpoint(self, thread_id: str) -> Optional[str]:
        """
        Get latest checkpoint for thread

        Args:
            thread_id: Thread ID

        Returns:
            Latest checkpoint ID or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT checkpoint_id FROM checkpoints
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (thread_id,))

        result = cursor.fetchone()
        return result[0] if result else None

    def rollback(
        self,
        thread_id: str,
        steps: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback to previous state

        Args:
            thread_id: Thread ID
            steps: Number of steps to roll back

        Returns:
            Previous state or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT state FROM checkpoints
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 1 OFFSET ?
        """, (thread_id, steps))

        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None

    def branch_conversation(
        self,
        parent_checkpoint_id: str,
        new_state: Dict[str, Any],
        new_thread_id: Optional[str] = None
    ) -> str:
        """
        Create branch from checkpoint (what-if scenario)

        Args:
            parent_checkpoint_id: Checkpoint to branch from
            new_state: New state for branch
            new_thread_id: Optional new thread ID

        Returns:
            New checkpoint ID
        """
        # Get parent checkpoint
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT thread_id FROM checkpoints WHERE checkpoint_id = ?
        """, (parent_checkpoint_id,))

        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Parent checkpoint {parent_checkpoint_id} not found")

        parent_thread_id = result[0]

        # Create branch
        if new_thread_id is None:
            import hashlib
            new_thread_id = f"{parent_thread_id}_branch_{hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:8]}"

        return self.save_checkpoint(
            thread_id=new_thread_id,
            state=new_state,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata={"branched_from": parent_checkpoint_id}
        )

    def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        List checkpoints for thread

        Args:
            thread_id: Thread ID
            limit: Maximum checkpoints to return

        Returns:
            List of checkpoint dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT checkpoint_id, created_at, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (thread_id, limit))

        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append({
                "checkpoint_id": row[0],
                "created_at": row[1],
                "metadata": json.loads(row[2])
            })

        return checkpoints

    def get_statistics(self) -> Dict:
        """Get checkpoint statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total checkpoints
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        stats['total_checkpoints'] = cursor.fetchone()[0]

        # Unique threads
        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        stats['unique_threads'] = cursor.fetchone()[0]

        # Branches
        cursor.execute("SELECT COUNT(*) FROM checkpoints WHERE parent_checkpoint_id IS NOT NULL")
        stats['branches'] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    print("="*70)
    print("LangGraph Checkpoint Manager Demo")
    print("="*70)

    manager = CheckpointManager()

    # Create conversation with checkpoints
    thread_id = "demo_conversation_1"

    # Checkpoint 1
    state1 = {"messages": ["Hello"], "context": {}}
    ckpt1 = manager.save_checkpoint(thread_id, state1)
    print(f"\n✓ Checkpoint 1: {ckpt1}")

    # Checkpoint 2
    state2 = {"messages": ["Hello", "How can I help?"], "context": {"user": "demo"}}
    ckpt2 = manager.save_checkpoint(thread_id, state2)
    print(f"✓ Checkpoint 2: {ckpt2}")

    # Checkpoint 3
    state3 = {"messages": ["Hello", "How can I help?", "Optimize SQL"], "context": {"user": "demo", "topic": "sql"}}
    ckpt3 = manager.save_checkpoint(thread_id, state3)
    print(f"✓ Checkpoint 3: {ckpt3}")

    # Load latest
    latest_id = manager.get_latest_checkpoint(thread_id)
    latest_state = manager.load_checkpoint(latest_id)
    print(f"\n✓ Latest checkpoint: {len(latest_state['messages'])} messages")

    # Rollback
    previous_state = manager.rollback(thread_id, steps=1)
    print(f"✓ Rolled back: {len(previous_state['messages'])} messages")

    # Branch conversation
    branch_state = {"messages": ["Hello", "Different question"], "context": {}}
    branch_id = manager.branch_conversation(ckpt1, branch_state)
    print(f"✓ Created branch: {branch_id}")

    # List checkpoints
    checkpoints = manager.list_checkpoints(thread_id)
    print(f"\n✓ Thread has {len(checkpoints)} checkpoints")

    # Statistics
    stats = manager.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    manager.close()
