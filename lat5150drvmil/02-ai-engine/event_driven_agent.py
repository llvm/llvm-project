#!/usr/bin/env python3
"""
Event-Driven Agent Architecture
Based on ai-that-works Episode #30: "Treat agent interactions as an event log"

Key Principles:
- Immutable event log (never mutate state directly)
- State projection from events
- Temporal reasoning and replay capability
- Audit trail for compliance

Event Types:
- UserInput: User messages/commands
- LLMChunk: Streaming LLM tokens
- LLMComplete: Full LLM response
- ToolCall: Agent invoking a tool
- ToolResult: Tool execution result
- Interrupt: User interruption/cancellation
- UIAction: UI-driven actions
- StateChange: Explicit state changes
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json
import uuid
import sqlite3
from collections import defaultdict


class EventType(Enum):
    """Types of events in the agent interaction log"""
    USER_INPUT = "user_input"
    LLM_CHUNK = "llm_chunk"
    LLM_COMPLETE = "llm_complete"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INTERRUPT = "interrupt"
    UI_ACTION = "ui_action"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    METADATA = "metadata"


@dataclass
class AgentEvent:
    """
    Immutable event in the agent interaction log

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        timestamp: When the event occurred (UTC)
        data: Event payload
        metadata: Optional metadata (tags, context, etc.)
        session_id: Session identifier for grouping
        parent_event_id: Parent event (for causality)
    """
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    parent_event_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "parent_event_id": self.parent_event_id
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'AgentEvent':
        """Deserialize event from dictionary"""
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            data=d["data"],
            session_id=d["session_id"],
            metadata=d.get("metadata"),
            parent_event_id=d.get("parent_event_id")
        )


class EventStore:
    """
    Persistent storage for immutable event log

    Uses SQLite for durability and queryability.
    Supports:
    - Append-only writes
    - Temporal queries
    - Session-based retrieval
    - Event replay
    """

    def __init__(self, db_path: str = "agent_events.db"):
        """
        Initialize event store

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                parent_event_id TEXT,
                data TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON events(session_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)")
        self.conn.commit()

    def append(self, event: AgentEvent) -> str:
        """
        Append event to log (immutable)

        Args:
            event: Event to append

        Returns:
            Event ID
        """
        self.conn.execute("""
            INSERT INTO events (event_id, event_type, timestamp, session_id, parent_event_id, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.event_type.value,
            event.timestamp.isoformat(),
            event.session_id,
            event.parent_event_id,
            json.dumps(event.data),
            json.dumps(event.metadata) if event.metadata else None
        ))
        self.conn.commit()
        return event.event_id

    def get_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[AgentEvent]:
        """
        Query events from store

        Args:
            session_id: Filter by session
            event_type: Filter by event type
            since: Events after this time
            until: Events before this time
            limit: Maximum number of events

        Returns:
            List of events matching criteria
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        events = []

        for row in cursor.fetchall():
            event = AgentEvent(
                event_id=row[0],
                event_type=EventType(row[1]),
                timestamp=datetime.fromisoformat(row[2]),
                session_id=row[3],
                parent_event_id=row[4],
                data=json.loads(row[5]),
                metadata=json.loads(row[6]) if row[6] else None
            )
            events.append(event)

        return events

    def get_event(self, event_id: str) -> Optional[AgentEvent]:
        """Get single event by ID"""
        events = self.conn.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,)
        ).fetchone()

        if not events:
            return None

        return AgentEvent(
            event_id=events[0],
            event_type=EventType(events[1]),
            timestamp=datetime.fromisoformat(events[2]),
            session_id=events[3],
            parent_event_id=events[4],
            data=json.loads(events[5]),
            metadata=json.loads(events[6]) if events[6] else None
        )

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_events,
                MIN(timestamp) as first_event,
                MAX(timestamp) as last_event,
                event_type,
                COUNT(*) as count_by_type
            FROM events
            WHERE session_id = ?
            GROUP BY event_type
        """, (session_id,))

        rows = cursor.fetchall()
        if not rows:
            return {}

        total = rows[0][0]
        first = rows[0][1]
        last = rows[0][2]

        by_type = {}
        for row in rows:
            by_type[row[3]] = row[4]

        return {
            "total_events": total,
            "first_event": first,
            "last_event": last,
            "events_by_type": by_type
        }


@dataclass
class ProjectedState:
    """
    Current state projected from event history

    This is the "view" of the agent's state at a point in time,
    computed from the immutable event log.
    """
    session_id: str
    last_user_input: Optional[str] = None
    last_llm_response: Optional[str] = None
    tool_calls: List[Dict] = field(default_factory=list)
    interruptions: int = 0
    error_count: int = 0
    total_tokens: int = 0
    conversation_turns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventProjector:
    """
    Projects current state from event history

    This implements the "state projection" pattern:
    - Events are the source of truth
    - State is derived/computed from events
    - No mutable state to drift
    """

    def __init__(self, event_store: EventStore):
        """
        Initialize projector

        Args:
            event_store: Event store to project from
        """
        self.event_store = event_store

    def project(self, session_id: str) -> ProjectedState:
        """
        Project current state for a session

        Args:
            session_id: Session to project

        Returns:
            Current state derived from events
        """
        events = self.event_store.get_events(session_id=session_id)

        state = ProjectedState(session_id=session_id)

        for event in events:
            # Update state based on event type
            if event.event_type == EventType.USER_INPUT:
                state.last_user_input = event.data.get("content")
                state.conversation_turns += 1

            elif event.event_type == EventType.LLM_COMPLETE:
                state.last_llm_response = event.data.get("content")
                state.total_tokens += event.data.get("tokens", 0)

            elif event.event_type == EventType.TOOL_CALL:
                state.tool_calls.append({
                    "tool": event.data.get("tool_name"),
                    "timestamp": event.timestamp
                })

            elif event.event_type == EventType.INTERRUPT:
                state.interruptions += 1

            elif event.event_type == EventType.ERROR:
                state.error_count += 1

            elif event.event_type == EventType.METADATA:
                state.metadata.update(event.data)

        return state

    def replay(
        self,
        session_id: str,
        handler: Callable[[AgentEvent, ProjectedState], None]
    ):
        """
        Replay events with handler callback

        Useful for debugging, auditing, or rebuilding state.

        Args:
            session_id: Session to replay
            handler: Callback function(event, current_state)
        """
        events = self.event_store.get_events(session_id=session_id)
        state = ProjectedState(session_id=session_id)

        for event in events:
            handler(event, state)
            # Update state for next iteration
            # (handler can also modify state if needed)


class EventDrivenAgent:
    """
    Agent using event sourcing for state management

    Benefits:
    - Complete audit trail
    - Temporal queries (what was state at time T?)
    - Replay capability for debugging
    - No state drift
    - DSMIL compliance (immutable logs)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        event_store: Optional[EventStore] = None
    ):
        """
        Initialize event-driven agent

        Args:
            session_id: Session ID (generates if None)
            event_store: Event store (creates default if None)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.event_store = event_store or EventStore()
        self.projector = EventProjector(self.event_store)

    def log_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        metadata: Optional[Dict] = None,
        parent_event_id: Optional[str] = None
    ) -> str:
        """
        Log an event (append to immutable log)

        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata
            parent_event_id: Parent event for causality

        Returns:
            Event ID
        """
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
            session_id=self.session_id,
            metadata=metadata,
            parent_event_id=parent_event_id
        )

        return self.event_store.append(event)

    def get_state(self) -> ProjectedState:
        """Get current state (projected from events)"""
        return self.projector.project(self.session_id)

    def get_events(self, **kwargs) -> List[AgentEvent]:
        """Get events for this session"""
        return self.event_store.get_events(session_id=self.session_id, **kwargs)

    def replay(self, handler: Callable):
        """Replay all events with handler"""
        self.projector.replay(self.session_id, handler)


def main():
    """Demo usage"""
    print("=== Event-Driven Agent Demo ===\n")

    # Create agent
    agent = EventDrivenAgent(session_id="demo_session")

    # Log some events
    print("1. User asks a question...")
    agent.log_event(
        EventType.USER_INPUT,
        {"content": "What is event sourcing?"}
    )

    print("2. LLM responds...")
    agent.log_event(
        EventType.LLM_COMPLETE,
        {
            "content": "Event sourcing is a pattern where state changes are stored as immutable events.",
            "tokens": 150,
            "model": "claude-3"
        }
    )

    print("3. Tool call...")
    agent.log_event(
        EventType.TOOL_CALL,
        {
            "tool_name": "search_docs",
            "params": {"query": "event sourcing examples"}
        }
    )

    print("4. Tool result...")
    agent.log_event(
        EventType.TOOL_RESULT,
        {
            "tool_name": "search_docs",
            "result": ["Doc 1", "Doc 2", "Doc 3"]
        }
    )

    # Project current state
    print("\n=== Current State (Projected from Events) ===")
    state = agent.get_state()
    print(f"Session: {state.session_id}")
    print(f"Last user input: {state.last_user_input}")
    print(f"Last LLM response: {state.last_llm_response}")
    print(f"Tool calls: {len(state.tool_calls)}")
    print(f"Total tokens: {state.total_tokens}")
    print(f"Conversation turns: {state.conversation_turns}")

    # Replay events
    print("\n=== Event Replay ===")
    def replay_handler(event: AgentEvent, state: ProjectedState):
        print(f"[{event.timestamp.isoformat()}] {event.event_type.value}: {event.data.get('content', event.data)[:50]}...")

    agent.replay(replay_handler)

    # Get all events
    print("\n=== All Events ===")
    events = agent.get_events()
    print(f"Total events: {len(events)}")

    for event in events:
        print(f"- {event.event_type.value} @ {event.timestamp.isoformat()}")


if __name__ == "__main__":
    main()
