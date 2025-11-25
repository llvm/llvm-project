#!/usr/bin/env python3
"""
RAM Disk Database with SQLite Backup
High-speed in-memory database with persistent backup

Architecture:
- RAM disk: Ultra-fast access during runtime (tmpfs mount)
- SQLite backup: Persistent local storage (simple, fast)
- On startup: Load SQLite → RAM disk
- Periodic sync: RAM disk → SQLite (every 60s or on demand)
- On crash: Reload from SQLite backup

Advantages over PostgreSQL:
- Much simpler (no server process)
- Faster startup (direct file copy)
- Lightweight (single file)
- Easy to backup (just copy .db file)
- Native Python support (no dependencies)
"""

import os
import sys
import sqlite3
import shutil
import time
import threading
import atexit
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ConversationMessage:
    """Single conversation message"""
    id: Optional[int]
    session_id: str
    timestamp: datetime
    role: str  # "user", "assistant", "system"
    content: str
    model: str
    latency_ms: float
    hardware_backend: str


class RAMDiskDatabase:
    """
    High-performance RAM disk database with SQLite backup
    """

    def __init__(
        self,
        backup_path: str = None,
        ramdisk_path: str = "/dev/shm/lat5150_ai",
        sync_interval_seconds: int = 60,
        auto_sync: bool = True
    ):
        """
        Initialize RAM disk database

        Args:
            backup_path: Path to persistent SQLite backup
            ramdisk_path: Path to RAM disk mount point
            sync_interval_seconds: How often to sync to backup
            auto_sync: Enable automatic background sync
        """
        # Paths
        if backup_path is None:
            backup_path = os.path.join(
                os.path.dirname(__file__),
                "data",
                "conversation_history.db"
            )

        self.backup_path = Path(backup_path)
        self.ramdisk_path = Path(ramdisk_path)
        self.ramdisk_db_path = self.ramdisk_path / "conversation_history.db"

        # Ensure directories exist
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        self.ramdisk_path.mkdir(parents=True, exist_ok=True)

        # Check if RAM disk is available (tmpfs at /dev/shm)
        self.ramdisk_available = self._check_ramdisk()

        if self.ramdisk_available:
            print(f"✓ RAM disk available: {self.ramdisk_path}")
            self.active_db_path = self.ramdisk_db_path
        else:
            print(f"⚠️  RAM disk not available, using disk: {self.backup_path}")
            self.active_db_path = self.backup_path

        # Load from backup if exists
        if self.ramdisk_available and self.backup_path.exists():
            print(f"  Loading backup: {self.backup_path}")
            shutil.copy2(self.backup_path, self.ramdisk_db_path)
            print(f"  Loaded into RAM disk: {self.ramdisk_db_path}")

        # Connect to active database
        self.conn = sqlite3.connect(
            str(self.active_db_path),
            check_same_thread=False  # Allow multi-threaded access
        )
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Initialize schema
        self._init_schema()

        # Statistics
        self.stats = {
            "messages_stored": 0,
            "messages_retrieved": 0,
            "syncs_performed": 0,
            "last_sync_time": None,
            "using_ramdisk": self.ramdisk_available
        }

        # Auto-sync
        self.sync_interval = sync_interval_seconds
        self.auto_sync_enabled = auto_sync
        self.sync_thread = None

        if auto_sync and self.ramdisk_available:
            self._start_auto_sync()

        # Register cleanup on exit
        atexit.register(self.shutdown)

        print(f"✓ Database initialized: {self.active_db_path}")
        print(f"  Auto-sync: {'enabled' if auto_sync else 'disabled'} ({sync_interval_seconds}s interval)")

    def _check_ramdisk(self) -> bool:
        """Check if RAM disk mount point is available"""
        # /dev/shm is standard tmpfs RAM disk on Linux
        parent = self.ramdisk_path.parent
        if not parent.exists():
            return False

        # Check if it's actually tmpfs (RAM disk)
        try:
            import subprocess
            result = subprocess.run(
                ["df", "-T", str(parent)],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "tmpfs" in result.stdout
        except:
            # If we can't verify, assume it's available if /dev/shm exists
            return str(parent) == "/dev/shm"

    def _init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        # Conversation messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                latency_ms REAL,
                hardware_backend TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_timestamp
            ON conversation_messages(session_id, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON conversation_messages(created_at DESC)
        """)

        # Agent state table (for future use)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                agent_id TEXT PRIMARY KEY,
                state_data TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT
            )
        """)

        self.conn.commit()

    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str = None,
        latency_ms: float = 0,
        hardware_backend: str = "CPU"
    ) -> int:
        """
        Store conversation message

        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            model: Model used
            latency_ms: Response latency
            hardware_backend: Hardware used

        Returns:
            Message ID
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO conversation_messages
            (session_id, timestamp, role, content, model, latency_ms, hardware_backend)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            time.time(),
            role,
            content,
            model,
            latency_ms,
            hardware_backend
        ))

        self.conn.commit()
        self.stats["messages_stored"] += 1

        return cursor.lastrowid

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[ConversationMessage]:
        """
        Retrieve conversation history for session

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List of conversation messages
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM conversation_messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))

        messages = []
        for row in cursor.fetchall():
            msg = ConversationMessage(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=datetime.fromtimestamp(row["timestamp"]),
                role=row["role"],
                content=row["content"],
                model=row["model"],
                latency_ms=row["latency_ms"],
                hardware_backend=row["hardware_backend"]
            )
            messages.append(msg)

        self.stats["messages_retrieved"] += len(messages)

        # Return in chronological order
        return list(reversed(messages))

    def get_recent_messages(self, limit: int = 50) -> List[ConversationMessage]:
        """Get most recent messages across all sessions"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM conversation_messages
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        messages = []
        for row in cursor.fetchall():
            msg = ConversationMessage(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=datetime.fromtimestamp(row["timestamp"]),
                role=row["role"],
                content=row["content"],
                model=row["model"],
                latency_ms=row["latency_ms"],
                hardware_backend=row["hardware_backend"]
            )
            messages.append(msg)

        return messages

    def sync_to_backup(self) -> bool:
        """
        Sync RAM disk database to persistent backup

        Returns:
            True if sync successful
        """
        if not self.ramdisk_available:
            return True  # Already on disk

        try:
            # Close connection temporarily
            self.conn.close()

            # Copy to backup
            shutil.copy2(self.ramdisk_db_path, self.backup_path)

            # Reopen connection
            self.conn = sqlite3.connect(
                str(self.active_db_path),
                check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row

            self.stats["syncs_performed"] += 1
            self.stats["last_sync_time"] = time.time()

            return True

        except Exception as e:
            print(f"❌ Sync failed: {e}")
            # Reopen connection even if sync failed
            self.conn = sqlite3.connect(
                str(self.active_db_path),
                check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row
            return False

    def _auto_sync_loop(self):
        """Background thread for automatic sync"""
        while self.auto_sync_enabled:
            time.sleep(self.sync_interval)

            if self.auto_sync_enabled:
                success = self.sync_to_backup()
                if success:
                    print(f"  [Auto-sync] Backed up to: {self.backup_path}")

    def _start_auto_sync(self):
        """Start automatic sync thread"""
        self.sync_thread = threading.Thread(
            target=self._auto_sync_loop,
            daemon=True,
            name="AutoSyncThread"
        )
        self.sync_thread.start()

    def shutdown(self):
        """Shutdown database and perform final sync"""
        print("\n🔄 Shutting down database...")

        # Stop auto-sync
        self.auto_sync_enabled = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)

        # Final sync
        if self.ramdisk_available:
            print("  Performing final backup...")
            self.sync_to_backup()

        # Close connection
        if self.conn:
            self.conn.close()

        print("✓ Database shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = self.stats.copy()

        # Add database size
        if self.active_db_path.exists():
            stats["db_size_mb"] = self.active_db_path.stat().st_size / (1024 * 1024)

        # Add backup size
        if self.backup_path.exists():
            stats["backup_size_mb"] = self.backup_path.stat().st_size / (1024 * 1024)

        # Add message count
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversation_messages")
        stats["total_messages"] = cursor.fetchone()[0]

        return stats

    def clear_old_messages(self, days: int = 30) -> int:
        """
        Clear messages older than specified days

        Args:
            days: Delete messages older than this many days

        Returns:
            Number of messages deleted
        """
        cutoff_time = time.time() - (days * 86400)

        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM conversation_messages
            WHERE timestamp < ?
        """, (cutoff_time,))

        deleted = cursor.rowcount
        self.conn.commit()

        return deleted


def demo():
    """Demonstration of RAM disk database"""
    print("=" * 70)
    print(" RAM Disk Database Demo")
    print("=" * 70)
    print()

    # Create database
    db = RAMDiskDatabase(auto_sync=True, sync_interval_seconds=10)

    print("\nStoring test messages...")

    # Store some messages
    session_id = "demo_session_001"

    db.store_message(
        session_id=session_id,
        role="user",
        content="What is the weather like?",
        model="deepseek-coder",
        latency_ms=0,
        hardware_backend="CPU"
    )

    db.store_message(
        session_id=session_id,
        role="assistant",
        content="I don't have access to real-time weather data.",
        model="deepseek-coder",
        latency_ms=245.3,
        hardware_backend="NPU"
    )

    print("✓ Stored 2 messages")

    # Retrieve history
    print(f"\nRetrieving conversation history for {session_id}...")
    messages = db.get_conversation_history(session_id)

    for msg in messages:
        print(f"  [{msg.role}] {msg.content[:50]}... (latency: {msg.latency_ms}ms, backend: {msg.hardware_backend})")

    # Statistics
    print("\n" + "=" * 70)
    print(" Database Statistics")
    print("=" * 70)

    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Database demo complete")
    print("\nNote: Auto-sync running in background every 10s")
    print("Press Ctrl+C to exit and perform final backup")

    try:
        # Keep alive to demonstrate auto-sync
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n")


if __name__ == "__main__":
    demo()
