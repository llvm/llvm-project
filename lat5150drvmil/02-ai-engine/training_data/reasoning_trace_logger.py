#!/usr/bin/env python3
"""
Reasoning Trace Logger for Training Data Generation

Logs agent reasoning steps to generate training data for:
- Supervised Fine-Tuning (SFT): Learn from successful reasoning patterns
- Reinforcement Learning (PPO/DPO): Reward successful traces
- Policy Learning: Learn when to retrieve more vs synthesize
- Error Analysis: Study failed traces to improve

Every complex query generates a reasoning trace that includes:
- Planning steps
- Retrieval actions
- Reflection decisions
- Critique evaluations
- Final synthesis

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class StepType(Enum):
    """Types of reasoning steps"""
    PLAN = "plan"
    RETRIEVE = "retrieve"
    REFLECT = "reflect"
    CRITIQUE = "critique"
    SYNTHESIZE = "synthesize"
    ERROR = "error"


@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_type: str  # StepType value
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: str
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        return asdict(self)


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query"""
    trace_id: str
    query: str
    steps: List[ReasoningStep]
    final_answer: Optional[str]
    success: bool
    quality_score: float  # 0.0-1.0
    user_feedback: Optional[str] = None
    user_rating: Optional[int] = None  # 1-5 stars
    total_duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        data = asdict(self)
        data['steps'] = [step.to_dict() if isinstance(step, ReasoningStep) else step for step in self.steps]
        return data


class ReasoningTraceLogger:
    """
    Log and manage reasoning traces for training data generation

    Usage:
        logger = ReasoningTraceLogger()

        # Start new trace
        trace_id = logger.start_trace(query="How do I optimize SQL?")

        # Log steps
        logger.log_step(trace_id, StepType.PLAN, "decompose_query", {...}, {...})
        logger.log_step(trace_id, StepType.RETRIEVE, "semantic_search", {...}, {...})

        # End trace
        logger.end_trace(trace_id, answer="...", success=True, quality=0.92)

        # Add user feedback
        logger.add_feedback(trace_id, rating=5, feedback="Very helpful!")

        # Export training data
        sft_data = logger.export_sft_training_data(min_quality=0.8)
    """

    def __init__(self, db_path: str = "~/.rag_index/reasoning_traces.db"):
        """
        Initialize reasoning trace logger

        Args:
            db_path: Path to SQLite database for traces
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()

        # In-memory trace buffer
        self.active_traces = {}

    def _init_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Traces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                final_answer TEXT,
                success INTEGER,
                quality_score REAL,
                user_feedback TEXT,
                user_rating INTEGER,
                total_duration_ms REAL,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Steps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT,
                step_type TEXT,
                action TEXT,
                input_data TEXT,
                output_data TEXT,
                timestamp TEXT,
                duration_ms REAL,
                success INTEGER,
                error_message TEXT,
                FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
            )
        """)

        # Indices for queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_quality ON traces(quality_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_success ON traces(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_rating ON traces(user_rating)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_type ON steps(step_type)")

        self.conn.commit()

    def start_trace(self, query: str, metadata: Optional[Dict] = None) -> str:
        """
        Start new reasoning trace

        Args:
            query: User query
            metadata: Optional metadata (e.g., user_id, session_id)

        Returns:
            Trace ID
        """
        import hashlib
        trace_id = hashlib.sha256(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        trace = ReasoningTrace(
            trace_id=trace_id,
            query=query,
            steps=[],
            final_answer=None,
            success=False,
            quality_score=0.0,
            metadata=metadata or {}
        )

        self.active_traces[trace_id] = trace
        print(f"Started trace: {trace_id}")
        return trace_id

    def log_step(
        self,
        trace_id: str,
        step_type: StepType,
        action: str,
        input_data: Dict,
        output_data: Dict,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Log a reasoning step

        Args:
            trace_id: Trace ID from start_trace()
            step_type: Type of step (StepType enum)
            action: Action name (e.g., "semantic_search", "decompose_query")
            input_data: Input parameters
            output_data: Step outputs
            duration_ms: Step duration in milliseconds
            success: Whether step succeeded
            error_message: Error message if failed
        """
        if trace_id not in self.active_traces:
            print(f"Warning: Trace {trace_id} not found. Call start_trace() first.")
            return

        step = ReasoningStep(
            step_type=step_type.value,
            action=action,
            input_data=input_data,
            output_data=output_data,
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )

        self.active_traces[trace_id].steps.append(step)

    def end_trace(
        self,
        trace_id: str,
        answer: Optional[str],
        success: bool,
        quality: float = 0.5,
        total_duration_ms: Optional[float] = None
    ):
        """
        End reasoning trace and save to database

        Args:
            trace_id: Trace ID
            answer: Final answer generated
            success: Whether task completed successfully
            quality: Quality score 0.0-1.0 (default: 0.5)
            total_duration_ms: Total execution time
        """
        if trace_id not in self.active_traces:
            print(f"Warning: Trace {trace_id} not found")
            return

        trace = self.active_traces[trace_id]
        trace.final_answer = answer
        trace.success = success
        trace.quality_score = quality
        trace.total_duration_ms = total_duration_ms

        # Save to database
        self._save_trace(trace)

        # Remove from active traces
        del self.active_traces[trace_id]

        print(f"Ended trace: {trace_id} (success={success}, quality={quality:.2f})")

    def add_feedback(
        self,
        trace_id: str,
        rating: Optional[int] = None,
        feedback: Optional[str] = None
    ):
        """
        Add user feedback to trace

        Args:
            trace_id: Trace ID
            rating: User rating 1-5 stars
            feedback: Text feedback
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE traces
            SET user_rating = ?, user_feedback = ?, updated_at = ?
            WHERE trace_id = ?
        """, (rating, feedback, datetime.now().isoformat(), trace_id))
        self.conn.commit()

        print(f"Added feedback to trace: {trace_id} (rating={rating})")

    def _save_trace(self, trace: ReasoningTrace):
        """Save trace and steps to database"""
        cursor = self.conn.cursor()

        # Save trace
        cursor.execute("""
            INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.trace_id,
            trace.query,
            trace.final_answer,
            1 if trace.success else 0,
            trace.quality_score,
            trace.user_feedback,
            trace.user_rating,
            trace.total_duration_ms,
            json.dumps(trace.metadata),
            trace.created_at,
            datetime.now().isoformat()
        ))

        # Save steps
        for step in trace.steps:
            cursor.execute("""
                INSERT INTO steps (trace_id, step_type, action, input_data, output_data,
                                 timestamp, duration_ms, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                step.step_type,
                step.action,
                json.dumps(step.input_data),
                json.dumps(step.output_data),
                step.timestamp,
                step.duration_ms,
                1 if step.success else 0,
                step.error_message
            ))

        self.conn.commit()

    def export_sft_training_data(
        self,
        min_quality: float = 0.8,
        min_rating: Optional[int] = 4,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Export successful traces as SFT training data

        Args:
            min_quality: Minimum quality score (0.0-1.0)
            min_rating: Minimum user rating (1-5)
            limit: Maximum number of examples

        Returns:
            List of training examples in format:
            {"query": "...", "reasoning_steps": [...], "answer": "..."}
        """
        cursor = self.conn.cursor()

        query = """
            SELECT trace_id, query, final_answer, quality_score, user_rating
            FROM traces
            WHERE success = 1 AND quality_score >= ?
        """
        params = [min_quality]

        if min_rating:
            query += " AND (user_rating IS NULL OR user_rating >= ?)"
            params.append(min_rating)

        query += " ORDER BY quality_score DESC, user_rating DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        traces = cursor.fetchall()

        training_data = []
        for trace_id, query_text, answer, quality, rating in traces:
            # Get steps
            cursor.execute("""
                SELECT step_type, action, input_data, output_data
                FROM steps
                WHERE trace_id = ?
                ORDER BY step_id
            """, (trace_id,))
            steps = cursor.fetchall()

            training_data.append({
                "query": query_text,
                "reasoning_steps": [
                    {
                        "type": st,
                        "action": act,
                        "input": json.loads(inp),
                        "output": json.loads(out)
                    }
                    for st, act, inp, out in steps
                ],
                "answer": answer,
                "quality_score": quality,
                "user_rating": rating
            })

        print(f"Exported {len(training_data)} SFT training examples")
        return training_data

    def get_statistics(self) -> Dict:
        """Get trace statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total traces
        cursor.execute("SELECT COUNT(*) FROM traces")
        stats['total_traces'] = cursor.fetchone()[0]

        # Success rate
        cursor.execute("SELECT AVG(success) FROM traces")
        stats['success_rate'] = cursor.fetchone()[0] or 0.0

        # Average quality
        cursor.execute("SELECT AVG(quality_score) FROM traces WHERE success = 1")
        stats['avg_quality'] = cursor.fetchone()[0] or 0.0

        # Rating distribution
        cursor.execute("""
            SELECT user_rating, COUNT(*)
            FROM traces
            WHERE user_rating IS NOT NULL
            GROUP BY user_rating
        """)
        stats['rating_distribution'] = dict(cursor.fetchall())

        # Step type distribution
        cursor.execute("""
            SELECT step_type, COUNT(*)
            FROM steps
            GROUP BY step_type
        """)
        stats['step_distribution'] = dict(cursor.fetchall())

        # High quality traces (for SFT)
        cursor.execute("SELECT COUNT(*) FROM traces WHERE quality_score >= 0.8 AND success = 1")
        stats['high_quality_traces'] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("Reasoning Trace Logger Demo")
    print("="*60)

    logger = ReasoningTraceLogger()

    # Example trace
    trace_id = logger.start_trace(
        query="How do I optimize PostgreSQL queries?",
        metadata={"user_id": "demo_user"}
    )

    # Log reasoning steps
    logger.log_step(
        trace_id,
        StepType.PLAN,
        "decompose_query",
        input_data={"query": "How do I optimize PostgreSQL queries?"},
        output_data={"sub_queries": ["indexing strategies", "EXPLAIN ANALYZE", "connection pooling"]},
        duration_ms=50.0
    )

    logger.log_step(
        trace_id,
        StepType.RETRIEVE,
        "semantic_search",
        input_data={"query": "indexing strategies", "n_results": 5},
        output_data={"documents": ["B-tree indexes...", "GIN indexes..."]},
        duration_ms=120.0
    )

    logger.log_step(
        trace_id,
        StepType.REFLECT,
        "evaluate_evidence",
        input_data={"evidence_count": 5},
        output_data={"decision": "sufficient", "confidence": 0.9},
        duration_ms=30.0
    )

    logger.log_step(
        trace_id,
        StepType.SYNTHESIZE,
        "generate_answer",
        input_data={"documents": 5, "query": "..."},
        output_data={"answer": "To optimize PostgreSQL queries: 1. Use appropriate indexes..."},
        duration_ms=200.0
    )

    # End trace
    logger.end_trace(
        trace_id,
        answer="To optimize PostgreSQL queries: 1. Use appropriate indexes 2. Analyze with EXPLAIN 3. Use connection pooling",
        success=True,
        quality=0.92,
        total_duration_ms=400.0
    )

    # Add user feedback
    logger.add_feedback(trace_id, rating=5, feedback="Very comprehensive answer!")

    # Show statistics
    stats = logger.get_statistics()
    print("\nTrace Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export SFT data
    sft_data = logger.export_sft_training_data(min_quality=0.8)
    print(f"\nExported {len(sft_data)} SFT training examples")

    logger.close()
