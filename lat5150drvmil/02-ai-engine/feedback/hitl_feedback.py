#!/usr/bin/env python3
"""
HITL (Human-in-the-Loop) Feedback Collector

Collects user feedback for AI improvement through:
- Thumbs up/down ratings
- Text corrections
- Quality ratings (1-5 stars)
- Explicit preferences (A vs B)

Feedback is used for:
- DPO (Direct Preference Optimization) training
- SFT (Supervised Fine-Tuning) with corrected data
- Quality metrics and A/B testing

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass, asdict


class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    RATING = "rating"
    PREFERENCE = "preference"  # Chose A over B


@dataclass
class Feedback:
    """User feedback record"""
    feedback_id: str
    query: str
    response: str
    feedback_type: str
    rating: Optional[int] = None  # 1-5 stars
    correction: Optional[str] = None  # Corrected response
    comment: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


class HITLFeedbackCollector:
    """
    Collect and manage user feedback for AI improvement

    Usage:
        collector = HITLFeedbackCollector()

        # Collect feedback
        collector.thumbs_up(query, response)
        collector.thumbs_down(query, response, comment="Too verbose")
        collector.correction(query, wrong_response, correct_response)
        collector.rating(query, response, rating=4)

        # Generate DPO training data
        dpo_pairs = collector.export_dpo_pairs(min_examples=100)
    """

    def __init__(self, db_path: str = "~/.rag_index/hitl_feedback.db"):
        """
        Initialize feedback collector

        Args:
            db_path: Path to SQLite database for feedback
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()

    def _init_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                correction TEXT,
                comment TEXT,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)

        # Preference pairs table (for DPO)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_pairs (
                pair_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                chosen_response TEXT NOT NULL,
                rejected_response TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)

        # Indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")

        self.conn.commit()

    def _generate_id(self, prefix: str = "fb") -> str:
        """Generate unique feedback ID"""
        import hashlib
        return prefix + "_" + hashlib.sha256(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

    def thumbs_up(
        self,
        query: str,
        response: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record positive feedback (thumbs up)

        Args:
            query: User query
            response: AI response
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata

        Returns:
            Feedback ID
        """
        feedback = Feedback(
            feedback_id=self._generate_id("thumbs_up"),
            query=query,
            response=response,
            feedback_type=FeedbackType.THUMBS_UP.value,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

        self._save_feedback(feedback)
        print(f"✓ Thumbs up recorded: {feedback.feedback_id}")
        return feedback.feedback_id

    def thumbs_down(
        self,
        query: str,
        response: str,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record negative feedback (thumbs down)

        Args:
            query: User query
            response: AI response
            comment: Optional explanation of what was wrong
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata

        Returns:
            Feedback ID
        """
        feedback = Feedback(
            feedback_id=self._generate_id("thumbs_down"),
            query=query,
            response=response,
            feedback_type=FeedbackType.THUMBS_DOWN.value,
            comment=comment,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

        self._save_feedback(feedback)
        print(f"✗ Thumbs down recorded: {feedback.feedback_id}")
        return feedback.feedback_id

    def correction(
        self,
        query: str,
        wrong_response: str,
        correct_response: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record user correction (creates DPO preference pair)

        Args:
            query: User query
            wrong_response: Incorrect AI response
            correct_response: User's corrected response
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata

        Returns:
            Feedback ID
        """
        feedback = Feedback(
            feedback_id=self._generate_id("correction"),
            query=query,
            response=wrong_response,
            feedback_type=FeedbackType.CORRECTION.value,
            correction=correct_response,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

        self._save_feedback(feedback)

        # Also create preference pair for DPO
        self._save_preference_pair(
            query=query,
            chosen=correct_response,
            rejected=wrong_response,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )

        print(f"✓ Correction recorded: {feedback.feedback_id}")
        return feedback.feedback_id

    def rating(
        self,
        query: str,
        response: str,
        rating: int,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record star rating (1-5)

        Args:
            query: User query
            response: AI response
            rating: Rating 1-5 stars
            comment: Optional comment
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata

        Returns:
            Feedback ID
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5 stars")

        feedback = Feedback(
            feedback_id=self._generate_id("rating"),
            query=query,
            response=response,
            feedback_type=FeedbackType.RATING.value,
            rating=rating,
            comment=comment,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

        self._save_feedback(feedback)
        print(f"✓ Rating recorded: {rating}/5 stars ({feedback.feedback_id})")
        return feedback.feedback_id

    def preference(
        self,
        query: str,
        chosen_response: str,
        rejected_response: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record explicit preference (chose A over B)

        Args:
            query: User query
            chosen_response: Preferred response
            rejected_response: Non-preferred response
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata

        Returns:
            Pair ID
        """
        pair_id = self._save_preference_pair(
            query=query,
            chosen=chosen_response,
            rejected=rejected_response,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )

        print(f"✓ Preference recorded: {pair_id}")
        return pair_id

    def _save_feedback(self, feedback: Feedback):
        """Save feedback to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.feedback_id,
            feedback.query,
            feedback.response,
            feedback.feedback_type,
            feedback.rating,
            feedback.correction,
            feedback.comment,
            feedback.user_id,
            feedback.session_id,
            json.dumps(feedback.metadata),
            feedback.created_at
        ))
        self.conn.commit()

    def _save_preference_pair(
        self,
        query: str,
        chosen: str,
        rejected: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save preference pair for DPO training"""
        pair_id = self._generate_id("pair")

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO preference_pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair_id,
            query,
            chosen,
            rejected,
            user_id,
            session_id,
            json.dumps(metadata or {}),
            datetime.now().isoformat()
        ))
        self.conn.commit()

        return pair_id

    def export_dpo_pairs(
        self,
        min_examples: int = 100,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Export preference pairs for DPO training

        Args:
            min_examples: Minimum number of examples needed
            user_id: Filter by specific user

        Returns:
            List of preference pairs in format:
            {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        cursor = self.conn.cursor()

        query = "SELECT query, chosen_response, rejected_response FROM preference_pairs"
        params = []

        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)

        query += " ORDER BY created_at DESC"

        if min_examples:
            query += " LIMIT ?"
            params.append(min_examples)

        cursor.execute(query, params)
        pairs = cursor.fetchall()

        dpo_dataset = [
            {
                "prompt": query,
                "chosen": chosen,
                "rejected": rejected
            }
            for query, chosen, rejected in pairs
        ]

        print(f"Exported {len(dpo_dataset)} DPO preference pairs")
        return dpo_dataset

    def export_positive_examples(
        self,
        min_rating: int = 4,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Export positive examples for SFT training

        Args:
            min_rating: Minimum rating (1-5)
            limit: Maximum examples to return

        Returns:
            List of {"query": "...", "response": "..."} examples
        """
        cursor = self.conn.cursor()

        query = """
            SELECT query, response
            FROM feedback
            WHERE (feedback_type = ? OR (feedback_type = ? AND rating >= ?))
        """
        params = [FeedbackType.THUMBS_UP.value, FeedbackType.RATING.value, min_rating]

        query += " ORDER BY created_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        examples = cursor.fetchall()

        sft_dataset = [
            {"query": q, "response": r}
            for q, r in examples
        ]

        print(f"Exported {len(sft_dataset)} positive SFT examples")
        return sft_dataset

    def get_statistics(self) -> Dict:
        """Get feedback statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total feedback
        cursor.execute("SELECT COUNT(*) FROM feedback")
        stats['total_feedback'] = cursor.fetchone()[0]

        # Feedback type distribution
        cursor.execute("""
            SELECT feedback_type, COUNT(*)
            FROM feedback
            GROUP BY feedback_type
        """)
        stats['feedback_types'] = dict(cursor.fetchall())

        # Average rating
        cursor.execute("""
            SELECT AVG(rating)
            FROM feedback
            WHERE feedback_type = ?
        """, (FeedbackType.RATING.value,))
        stats['avg_rating'] = cursor.fetchone()[0] or 0.0

        # Total preference pairs
        cursor.execute("SELECT COUNT(*) FROM preference_pairs")
        stats['preference_pairs'] = cursor.fetchone()[0]

        # Positive vs negative
        cursor.execute("""
            SELECT COUNT(*)
            FROM feedback
            WHERE feedback_type IN (?, ?)
        """, (FeedbackType.THUMBS_UP.value, "rating"))
        positive = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM feedback
            WHERE feedback_type = ?
        """, (FeedbackType.THUMBS_DOWN.value,))
        negative = cursor.fetchone()[0]

        stats['positive_feedback'] = positive
        stats['negative_feedback'] = negative
        stats['satisfaction_rate'] = positive / (positive + negative) if (positive + negative) > 0 else 0.0

        return stats

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("HITL Feedback Collector Demo")
    print("="*60)

    collector = HITLFeedbackCollector()

    # Collect various types of feedback
    collector.thumbs_up(
        query="How do I use async/await in Python?",
        response="Use async def for coroutines and await to call them...",
        user_id="demo_user"
    )

    collector.thumbs_down(
        query="Explain quantum computing",
        response="Quantum computers use qubits...",
        comment="Too technical for a beginner",
        user_id="demo_user"
    )

    collector.correction(
        query="What is the capital of France?",
        wrong_response="The capital of France is Lyon.",
        correct_response="The capital of France is Paris.",
        user_id="demo_user"
    )

    collector.rating(
        query="Best practices for REST APIs",
        response="1. Use proper HTTP methods 2. Version your API 3. Use HTTPS...",
        rating=5,
        comment="Very comprehensive!",
        user_id="demo_user"
    )

    collector.preference(
        query="Explain recursion",
        chosen_response="Recursion is when a function calls itself. Base case prevents infinite loop...",
        rejected_response="Recursion = function calling function",
        user_id="demo_user"
    )

    # Show statistics
    stats = collector.get_statistics()
    print("\nFeedback Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export training data
    dpo_pairs = collector.export_dpo_pairs(min_examples=10)
    print(f"\nDPO pairs available: {len(dpo_pairs)}")

    sft_examples = collector.export_positive_examples(min_rating=4)
    print(f"Positive SFT examples: {len(sft_examples)}")

    collector.close()
