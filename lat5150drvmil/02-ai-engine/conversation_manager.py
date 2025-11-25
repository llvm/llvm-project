#!/usr/bin/env python3
"""
Conversation History Manager with Cross-Session Retrieval

Manages persistent conversation history with PostgreSQL backend.
Enables "remember our last conversation" functionality.

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from uuid import UUID, uuid4
import os


@dataclass
class Message:
    """Represents a single message"""
    id: str
    conversation_id: str
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    model: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    latency_ms: Optional[int] = None
    temperature: Optional[float] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class Conversation:
    """Represents a conversation session"""
    id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    archived: bool = False
    metadata: Optional[Dict] = None
    messages: Optional[List[Message]] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        if self.messages:
            data['messages'] = [m.to_dict() for m in self.messages]
        return data


class ConversationManager:
    """
    Manages conversation history with PostgreSQL persistence

    Features:
    - Create and manage conversations
    - Add messages with full metadata
    - Cross-session retrieval
    - Semantic search across conversations
    - Analytics and usage tracking
    """

    def __init__(self,
                 db_host: str = "localhost",
                 db_port: int = 5432,
                 db_name: str = "dsmil_ai",
                 db_user: str = "dsmil",
                 db_password: str = None):
        """
        Initialize conversation manager with database connection

        Args:
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password (or use env var DB_PASSWORD)
        """
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password or os.getenv('DB_PASSWORD', 'dsmil_secure_password')
        }

        # Create connection pool (min 1, max 10 connections)
        self.pool = ThreadedConnectionPool(1, 10, **self.db_config)

    def _get_conn(self):
        """Get connection from pool"""
        return self.pool.getconn()

    def _put_conn(self, conn):
        """Return connection to pool"""
        self.pool.putconn(conn)

    def create_conversation(self,
                          user_id: Optional[str] = None,
                          title: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> Conversation:
        """
        Create a new conversation

        Returns:
            Conversation object with generated ID
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get default user if none specified
                if not user_id:
                    cur.execute("SELECT id FROM users WHERE username = 'default' LIMIT 1")
                    result = cur.fetchone()
                    user_id = str(result['id']) if result else None

                cur.execute("""
                    INSERT INTO conversations (user_id, title, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id, user_id, title, summary, created_at, updated_at, archived, metadata
                """, (user_id, title, Json(metadata or {})))

                result = cur.fetchone()
                conn.commit()

                return Conversation(
                    id=str(result['id']),
                    user_id=str(result['user_id']) if result['user_id'] else None,
                    title=result['title'],
                    summary=result['summary'],
                    created_at=result['created_at'],
                    updated_at=result['updated_at'],
                    archived=result['archived'],
                    metadata=result['metadata']
                )
        finally:
            self._put_conn(conn)

    def add_message(self,
                   conversation_id: str,
                   role: str,
                   content: str,
                   model: Optional[str] = None,
                   tokens_input: Optional[int] = None,
                   tokens_output: Optional[int] = None,
                   latency_ms: Optional[int] = None,
                   temperature: Optional[float] = None,
                   metadata: Optional[Dict] = None) -> Message:
        """
        Add a message to a conversation

        Returns:
            Message object with generated ID
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO messages
                    (conversation_id, role, content, model, tokens_input, tokens_output,
                     latency_ms, temperature, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, conversation_id, role, content, model, tokens_input,
                              tokens_output, latency_ms, temperature, created_at, metadata
                """, (conversation_id, role, content, model, tokens_input, tokens_output,
                      latency_ms, temperature, Json(metadata or {})))

                result = cur.fetchone()

                # Update conversation's updated_at
                cur.execute("""
                    UPDATE conversations
                    SET updated_at = NOW()
                    WHERE id = %s
                """, (conversation_id,))

                conn.commit()

                return Message(
                    id=str(result['id']),
                    conversation_id=str(result['conversation_id']),
                    role=result['role'],
                    content=result['content'],
                    model=result['model'],
                    tokens_input=result['tokens_input'],
                    tokens_output=result['tokens_output'],
                    latency_ms=result['latency_ms'],
                    temperature=result['temperature'],
                    created_at=result['created_at'],
                    metadata=result['metadata']
                )
        finally:
            self._put_conn(conn)

    def get_conversation(self,
                        conversation_id: str,
                        include_messages: bool = True) -> Optional[Conversation]:
        """
        Get a conversation by ID

        Args:
            conversation_id: Conversation UUID
            include_messages: Whether to include all messages

        Returns:
            Conversation object or None if not found
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, user_id, title, summary, created_at, updated_at, archived, metadata
                    FROM conversations
                    WHERE id = %s
                """, (conversation_id,))

                result = cur.fetchone()
                if not result:
                    return None

                conversation = Conversation(
                    id=str(result['id']),
                    user_id=str(result['user_id']) if result['user_id'] else None,
                    title=result['title'],
                    summary=result['summary'],
                    created_at=result['created_at'],
                    updated_at=result['updated_at'],
                    archived=result['archived'],
                    metadata=result['metadata']
                )

                if include_messages:
                    cur.execute("""
                        SELECT id, conversation_id, role, content, model, tokens_input,
                               tokens_output, latency_ms, temperature, created_at, metadata
                        FROM messages
                        WHERE conversation_id = %s
                        ORDER BY created_at ASC
                    """, (conversation_id,))

                    messages = []
                    for msg in cur.fetchall():
                        messages.append(Message(
                            id=str(msg['id']),
                            conversation_id=str(msg['conversation_id']),
                            role=msg['role'],
                            content=msg['content'],
                            model=msg['model'],
                            tokens_input=msg['tokens_input'],
                            tokens_output=msg['tokens_output'],
                            latency_ms=msg['latency_ms'],
                            temperature=msg['temperature'],
                            created_at=msg['created_at'],
                            metadata=msg['metadata']
                        ))
                    conversation.messages = messages

                return conversation
        finally:
            self._put_conn(conn)

    def get_recent_conversations(self,
                                user_id: Optional[str] = None,
                                limit: int = 20,
                                include_archived: bool = False) -> List[Conversation]:
        """
        Get recent conversations for a user

        Args:
            user_id: User ID (None for all users)
            limit: Maximum number of conversations
            include_archived: Whether to include archived conversations

        Returns:
            List of Conversation objects
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT c.id, c.user_id, c.title, c.summary, c.created_at,
                           c.updated_at, c.archived, c.metadata,
                           COUNT(m.id) as message_count,
                           MAX(m.created_at) as last_message_at
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE 1=1
                """
                params = []

                if user_id:
                    query += " AND c.user_id = %s"
                    params.append(user_id)

                if not include_archived:
                    query += " AND c.archived = FALSE"

                query += """
                    GROUP BY c.id, c.user_id, c.title, c.summary, c.created_at,
                             c.updated_at, c.archived, c.metadata
                    ORDER BY COALESCE(MAX(m.created_at), c.updated_at) DESC
                    LIMIT %s
                """
                params.append(limit)

                cur.execute(query, params)

                conversations = []
                for row in cur.fetchall():
                    conversations.append(Conversation(
                        id=str(row['id']),
                        user_id=str(row['user_id']) if row['user_id'] else None,
                        title=row['title'],
                        summary=row['summary'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        archived=row['archived'],
                        metadata=row['metadata']
                    ))

                return conversations
        finally:
            self._put_conn(conn)

    def search_conversations(self,
                           query: str,
                           user_id: Optional[str] = None,
                           limit: int = 10) -> List[Conversation]:
        """
        Search conversations by content (full-text search)

        Args:
            query: Search query
            user_id: User ID filter (optional)
            limit: Maximum results

        Returns:
            List of matching conversations
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                search_query = """
                    SELECT DISTINCT c.id, c.user_id, c.title, c.summary,
                           c.created_at, c.updated_at, c.archived, c.metadata
                    FROM conversations c
                    JOIN messages m ON c.id = m.conversation_id
                    WHERE (m.content ILIKE %s OR c.title ILIKE %s OR c.summary ILIKE %s)
                """
                params = [f'%{query}%', f'%{query}%', f'%{query}%']

                if user_id:
                    search_query += " AND c.user_id = %s"
                    params.append(user_id)

                search_query += " ORDER BY c.updated_at DESC LIMIT %s"
                params.append(limit)

                cur.execute(search_query, params)

                conversations = []
                for row in cur.fetchall():
                    conversations.append(Conversation(
                        id=str(row['id']),
                        user_id=str(row['user_id']) if row['user_id'] else None,
                        title=row['title'],
                        summary=row['summary'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        archived=row['archived'],
                        metadata=row['metadata']
                    ))

                return conversations
        finally:
            self._put_conn(conn)

    def get_last_conversation(self, user_id: Optional[str] = None) -> Optional[Conversation]:
        """
        Get the most recent conversation (for "remember our last conversation")

        Args:
            user_id: User ID (None for default user)

        Returns:
            Most recent Conversation or None
        """
        conversations = self.get_recent_conversations(user_id=user_id, limit=1)
        return conversations[0] if conversations else None

    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE conversations
                    SET title = %s, updated_at = NOW()
                    WHERE id = %s
                """, (title, conversation_id))
                conn.commit()
        finally:
            self._put_conn(conn)

    def update_conversation_summary(self, conversation_id: str, summary: str):
        """Update conversation summary"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE conversations
                    SET summary = %s, updated_at = NOW()
                    WHERE id = %s
                """, (summary, conversation_id))
                conn.commit()
        finally:
            self._put_conn(conn)

    def archive_conversation(self, conversation_id: str):
        """Archive a conversation"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE conversations
                    SET archived = TRUE, updated_at = NOW()
                    WHERE id = %s
                """, (conversation_id,))
                conn.commit()
        finally:
            self._put_conn(conn)

    def get_statistics(self, user_id: Optional[str] = None) -> Dict:
        """
        Get conversation statistics

        Returns:
            Dict with stats (total_conversations, total_messages, avg_messages_per_conversation, etc.)
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT
                        COUNT(DISTINCT c.id) as total_conversations,
                        COUNT(m.id) as total_messages,
                        AVG(msg_counts.message_count) as avg_messages_per_conversation,
                        SUM(m.tokens_input + m.tokens_output) as total_tokens,
                        AVG(m.latency_ms) as avg_latency_ms
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    LEFT JOIN (
                        SELECT conversation_id, COUNT(*) as message_count
                        FROM messages
                        GROUP BY conversation_id
                    ) msg_counts ON c.id = msg_counts.conversation_id
                    WHERE 1=1
                """
                params = []

                if user_id:
                    query += " AND c.user_id = %s"
                    params.append(user_id)

                cur.execute(query, params)
                result = cur.fetchone()

                return {
                    'total_conversations': result['total_conversations'] or 0,
                    'total_messages': result['total_messages'] or 0,
                    'avg_messages_per_conversation': float(result['avg_messages_per_conversation'] or 0),
                    'total_tokens': result['total_tokens'] or 0,
                    'avg_latency_ms': float(result['avg_latency_ms'] or 0)
                }
        finally:
            self._put_conn(conn)

    def close(self):
        """Close database connection pool"""
        self.pool.closeall()


# Example usage and testing
if __name__ == "__main__":
    print("Conversation Manager Test")
    print("=" * 60)

    # Initialize (will fail if DB not set up - that's expected)
    try:
        manager = ConversationManager()

        # Create a conversation
        conv = manager.create_conversation(title="Test Conversation")
        print(f"Created conversation: {conv.id}")

        # Add messages
        msg1 = manager.add_message(
            conversation_id=conv.id,
            role="user",
            content="What is the context window size?",
            model="uncensored_code"
        )
        print(f"Added user message: {msg1.id}")

        msg2 = manager.add_message(
            conversation_id=conv.id,
            role="assistant",
            content="The context window is 8192 tokens.",
            model="uncensored_code",
            tokens_output=15,
            latency_ms=1250
        )
        print(f"Added assistant message: {msg2.id}")

        # Retrieve conversation
        retrieved = manager.get_conversation(conv.id, include_messages=True)
        print(f"\nRetrieved conversation with {len(retrieved.messages)} messages")

        # Get statistics
        stats = manager.get_statistics()
        print(f"\nStatistics: {stats}")

        # Close
        manager.close()

    except Exception as e:
        print(f"Error (expected if DB not set up): {e}")
