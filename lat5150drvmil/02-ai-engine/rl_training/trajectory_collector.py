#!/usr/bin/env python3
"""
Trajectory Collector for RL Training

Collects (state, action, reward, next_state) trajectories for:
- PPO training
- Policy evaluation
- Advantage estimation

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Trajectory:
    """Single trajectory (episode) of agent execution"""
    trajectory_id: str
    query: str
    states: List[Dict]
    actions: List[str]
    rewards: List[float]
    done: bool
    total_reward: float
    episode_length: int
    success: bool
    metadata: Dict[str, Any] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


class TrajectoryCollector:
    """
    Collect and manage RL trajectories

    Usage:
        collector = TrajectoryCollector()

        # Start episode
        traj_id = collector.start_trajectory(query="...")

        # Add steps
        collector.add_step(traj_id, state, action, reward, next_state)

        # End episode
        collector.end_trajectory(traj_id, success=True)

        # Get trajectories for training
        trajectories = collector.get_trajectories(min_reward=5.0)
    """

    def __init__(self, db_path: str = "~/.rag_index/rl_trajectories.db"):
        """
        Initialize trajectory collector

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()

        # In-memory active trajectories
        self.active_trajectories = {}

    def _init_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Trajectories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                query TEXT,
                states TEXT,
                actions TEXT,
                rewards TEXT,
                done INTEGER,
                total_reward REAL,
                episode_length INTEGER,
                success INTEGER,
                metadata TEXT,
                created_at TEXT
            )
        """)

        # Indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_total_reward ON trajectories(total_reward)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON trajectories(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episode_length ON trajectories(episode_length)")

        self.conn.commit()

    def start_trajectory(self, query: str, metadata: Optional[Dict] = None) -> str:
        """
        Start new trajectory

        Args:
            query: Initial query
            metadata: Optional metadata

        Returns:
            Trajectory ID
        """
        import hashlib
        traj_id = "traj_" + hashlib.sha256(
            f"{query}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        self.active_trajectories[traj_id] = {
            "query": query,
            "states": [],
            "actions": [],
            "rewards": [],
            "metadata": metadata or {}
        }

        return traj_id

    def add_step(
        self,
        trajectory_id: str,
        state: Dict,
        action: str,
        reward: float,
        next_state: Dict = None
    ):
        """
        Add step to trajectory

        Args:
            trajectory_id: Trajectory ID
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        if trajectory_id not in self.active_trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")

        traj = self.active_trajectories[trajectory_id]
        traj["states"].append(state)
        traj["actions"].append(action)
        traj["rewards"].append(reward)

        if next_state is not None:
            # Store next_state for last step
            traj["next_state"] = next_state

    def end_trajectory(
        self,
        trajectory_id: str,
        success: bool,
        done: bool = True
    ):
        """
        End trajectory and save to database

        Args:
            trajectory_id: Trajectory ID
            success: Whether episode succeeded
            done: Whether episode is complete
        """
        if trajectory_id not in self.active_trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")

        traj = self.active_trajectories[trajectory_id]

        # Calculate total reward
        total_reward = sum(traj["rewards"])

        # Create Trajectory object
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            query=traj["query"],
            states=traj["states"],
            actions=traj["actions"],
            rewards=traj["rewards"],
            done=done,
            total_reward=total_reward,
            episode_length=len(traj["actions"]),
            success=success,
            metadata=traj["metadata"]
        )

        # Save to database
        self._save_trajectory(trajectory)

        # Remove from active
        del self.active_trajectories[trajectory_id]

        print(f"Trajectory {trajectory_id} completed: reward={total_reward:.2f}, success={success}")

    def _save_trajectory(self, trajectory: Trajectory):
        """Save trajectory to database"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trajectory.trajectory_id,
            trajectory.query,
            json.dumps(trajectory.states),
            json.dumps(trajectory.actions),
            json.dumps(trajectory.rewards),
            1 if trajectory.done else 0,
            trajectory.total_reward,
            trajectory.episode_length,
            1 if trajectory.success else 0,
            json.dumps(trajectory.metadata),
            trajectory.created_at
        ))

        self.conn.commit()

    def get_trajectories(
        self,
        min_reward: float = None,
        min_length: int = None,
        success_only: bool = False,
        limit: int = None
    ) -> List[Trajectory]:
        """
        Get trajectories for training

        Args:
            min_reward: Minimum total reward
            min_length: Minimum episode length
            success_only: Only successful episodes
            limit: Maximum number of trajectories

        Returns:
            List of Trajectory objects
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM trajectories WHERE 1=1"
        params = []

        if min_reward is not None:
            query += " AND total_reward >= ?"
            params.append(min_reward)

        if min_length is not None:
            query += " AND episode_length >= ?"
            params.append(min_length)

        if success_only:
            query += " AND success = 1"

        query += " ORDER BY total_reward DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        trajectories = []
        for row in rows:
            trajectories.append(Trajectory(
                trajectory_id=row[0],
                query=row[1],
                states=json.loads(row[2]),
                actions=json.loads(row[3]),
                rewards=json.loads(row[4]),
                done=bool(row[5]),
                total_reward=row[6],
                episode_length=row[7],
                success=bool(row[8]),
                metadata=json.loads(row[9]),
                created_at=row[10]
            ))

        return trajectories

    def get_statistics(self) -> Dict:
        """Get trajectory statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total trajectories
        cursor.execute("SELECT COUNT(*) FROM trajectories")
        stats['total_trajectories'] = cursor.fetchone()[0]

        # Success rate
        cursor.execute("SELECT AVG(success) FROM trajectories")
        stats['success_rate'] = cursor.fetchone()[0] or 0.0

        # Average reward
        cursor.execute("SELECT AVG(total_reward) FROM trajectories")
        stats['avg_reward'] = cursor.fetchone()[0] or 0.0

        # Average episode length
        cursor.execute("SELECT AVG(episode_length) FROM trajectories")
        stats['avg_episode_length'] = cursor.fetchone()[0] or 0.0

        # Best trajectory
        cursor.execute("SELECT total_reward FROM trajectories ORDER BY total_reward DESC LIMIT 1")
        result = cursor.fetchone()
        stats['best_reward'] = result[0] if result else 0.0

        return stats

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    print("="*70)
    print("Trajectory Collector Demo")
    print("="*70)

    collector = TrajectoryCollector()

    # Simulate trajectory collection
    traj_id = collector.start_trajectory("Optimize PostgreSQL performance")

    # Add steps
    state1 = {"iteration": 1, "documents": 0, "relevance": 0.0}
    state2 = {"iteration": 2, "documents": 5, "relevance": 0.7}
    state3 = {"iteration": 3, "documents": 5, "relevance": 0.9}

    collector.add_step(traj_id, state1, "retrieve", reward=2.0, next_state=state2)
    collector.add_step(traj_id, state2, "refine", reward=3.0, next_state=state3)
    collector.add_step(traj_id, state3, "synthesize", reward=10.0)

    # End trajectory
    collector.end_trajectory(traj_id, success=True)

    # Get statistics
    stats = collector.get_statistics()
    print(f"\nTrajectory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get best trajectories
    trajectories = collector.get_trajectories(min_reward=10.0, limit=5)
    print(f"\nHigh-reward trajectories: {len(trajectories)}")

    collector.close()
