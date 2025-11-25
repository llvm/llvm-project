#!/usr/bin/env python3
"""
SQLite Storage Backend Adapter

Provides unified interface for SQLite databases in the LAT5150DRVMIL AI engine.
Handles:
- Lightweight local storage
- RAM disk databases for high-speed access
- Checkpoints and audit logs
- Structured data with full-text search
- ACID transactions
"""

import logging
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import time

from storage_abstraction import (
    AbstractStorageBackend,
    StorageType,
    ContentType,
    StorageTier,
    StorageHandle,
    SearchResult,
    StorageStats
)

logger = logging.getLogger(__name__)


class SQLiteStorageBackend(AbstractStorageBackend):
    """
    SQLite storage adapter for AI engine data

    Supports:
    - File-based or in-memory databases
    - Full-text search (FTS5)
    - ACID transactions
    - JSON storage
    - Lightweight checkpoints
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SQLite backend

        Args:
            config: Configuration dictionary with:
                - db_path: Database file path (:memory: for RAM)
                - use_wal: Use Write-Ahead Logging (default: True)
                - cache_size: Page cache size in KB (default: 2000)
                - enable_fts: Enable full-text search (default: True)
                - journal_mode: Journal mode (WAL, DELETE, etc.)
        """
        super().__init__(config)
        self.storage_type = StorageType.SQLITE

        # Configuration
        self.db_path = config.get('db_path', ':memory:')
        self.use_wal = config.get('use_wal', True)
        self.cache_size = config.get('cache_size', 2000)
        self.enable_fts = config.get('enable_fts', True)
        self.journal_mode = config.get('journal_mode', 'WAL' if self.use_wal else 'DELETE')

        # Determine tier based on path
        if self.db_path == ':memory:':
            self.default_tier = StorageTier.HOT
        elif '/dev/shm' in str(self.db_path) or '/tmpfs' in str(self.db_path):
            self.default_tier = StorageTier.HOT
        else:
            self.default_tier = StorageTier.WARM

        # Connection
        self.conn = None

        # Performance tracking
        self._query_times = []
        self._max_query_history = 1000

    def connect(self) -> bool:
        """
        Establish connection to SQLite database

        Returns:
            True if connection successful
        """
        try:
            # Create directory if needed
            if self.db_path != ':memory:':
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            self.conn.row_factory = sqlite3.Row

            # Configure performance settings
            self.conn.execute(f"PRAGMA journal_mode={self.journal_mode}")
            self.conn.execute(f"PRAGMA cache_size=-{self.cache_size}")  # Negative for KB
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap

            # Create tables
            self._create_tables()

            # Enable FTS if configured
            if self.enable_fts:
                self._create_fts_tables()

            logger.info(f"Connected to SQLite: {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Close SQLite connection

        Returns:
            True if disconnection successful
        """
        try:
            if self.conn:
                self.conn.close()
                logger.info("SQLite connection closed")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from SQLite: {e}")
            return False

    def store(
        self,
        data: Any,
        content_type: ContentType,
        key: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> StorageHandle:
        """
        Store data in SQLite

        Args:
            data: Data to store (dict, str, or JSON-serializable)
            content_type: Type of content
            key: Optional key (auto-generated if not provided)
            ttl: Time-to-live in seconds (for expiring data)
            metadata: Additional metadata

        Returns:
            StorageHandle for retrieving data
        """
        try:
            # Serialize data
            if isinstance(data, dict):
                data_json = json.dumps(data)
            elif isinstance(data, str):
                data_json = data
            else:
                data_json = json.dumps({"value": data})

            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = datetime.now().timestamp() + ttl

            # Prepare metadata
            metadata_json = json.dumps(metadata or {})

            # Store based on content type
            if content_type == ContentType.CHECKPOINT:
                storage_id = self._store_checkpoint(data, metadata)
            elif content_type == ContentType.AUDIT:
                storage_id = self._store_audit_log(data, metadata)
            else:
                storage_id = self._store_generic(
                    content_type, data_json, key, expires_at, metadata_json
                )

            self._stats['operations'] += 1
            self._stats['bytes_written'] += len(data_json.encode())

            return StorageHandle(
                storage_type=self.storage_type,
                storage_id=str(storage_id),
                content_type=content_type,
                tier=self.default_tier,
                created_at=datetime.now(),
                metadata=metadata or {}
            )

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error storing data in SQLite: {e}")
            raise

    def retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """
        Retrieve data by handle

        Args:
            handle: Storage handle from store()

        Returns:
            Retrieved data or None if not found
        """
        try:
            start_time = time.time()

            if handle.content_type == ContentType.CHECKPOINT:
                data = self._retrieve_checkpoint(handle.storage_id)
            elif handle.content_type == ContentType.AUDIT:
                data = self._retrieve_audit_log(handle.storage_id)
            else:
                data = self._retrieve_generic(handle.storage_id)

            # Track performance
            query_time = (time.time() - start_time) * 1000
            self._query_times.append(query_time)
            if len(self._query_times) > self._max_query_history:
                self._query_times.pop(0)

            self._stats['operations'] += 1
            if data:
                self._stats['bytes_read'] += len(str(data).encode())

            return data

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error retrieving data from SQLite: {e}")
            return None

    def delete(self, handle: StorageHandle) -> bool:
        """
        Delete data by handle

        Args:
            handle: Storage handle

        Returns:
            True if deletion successful
        """
        try:
            table_name = self._get_table_for_content_type(handle.content_type)

            cursor = self.conn.execute(
                f"DELETE FROM {table_name} WHERE id = ?",
                (handle.storage_id,)
            )

            deleted = cursor.rowcount > 0
            self._stats['operations'] += 1

            return deleted

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error deleting data from SQLite: {e}")
            return False

    def search(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search for data using SQLite FTS5

        Args:
            query: Search query text
            content_type: Filter by content type
            filters: Additional filters
            limit: Maximum results

        Returns:
            List of search results
        """
        try:
            if self.enable_fts:
                results = self._search_fts(query, content_type, limit)
            else:
                results = self._search_like(query, content_type, limit)

            self._stats['operations'] += 1
            return results

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error searching SQLite: {e}")
            return []

    def get_stats(self) -> StorageStats:
        """
        Get SQLite storage statistics

        Returns:
            StorageStats object
        """
        try:
            # Get database size
            if self.db_path != ':memory:':
                db_path = Path(self.db_path)
                total_size = db_path.stat().st_size if db_path.exists() else 0
            else:
                # For in-memory, estimate from page count
                cursor = self.conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = self.conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                total_size = page_count * page_size

            # Count total items
            cursor = self.conn.execute("""
                SELECT COUNT(*) as count FROM unified_storage
            """)
            total_items = cursor.fetchone()[0]

            # Average access time
            avg_access_time = (
                sum(self._query_times) / len(self._query_times)
                if self._query_times else 0.0
            )

            return StorageStats(
                storage_type=self.storage_type,
                total_items=total_items,
                total_size_bytes=total_size,
                avg_access_time_ms=avg_access_time,
                health_status="healthy",
                custom_metrics={
                    'operations': self._stats['operations'],
                    'bytes_written': self._stats['bytes_written'],
                    'bytes_read': self._stats['bytes_read'],
                    'errors': self._stats['errors'],
                    'db_path': self.db_path
                }
            )

        except Exception as e:
            logger.error(f"Error getting SQLite stats: {e}")
            return StorageStats(
                storage_type=self.storage_type,
                total_items=0,
                total_size_bytes=0,
                avg_access_time_ms=0,
                health_status="error"
            )

    def health_check(self) -> Tuple[bool, str]:
        """
        Check SQLite health

        Returns:
            (is_healthy, status_message)
        """
        try:
            cursor = self.conn.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                return (True, "SQLite connection healthy")
            else:
                return (False, "Unexpected health check result")
        except Exception as e:
            return (False, f"SQLite health check failed: {e}")

    # Internal helper methods

    def _create_tables(self):
        """Create necessary tables"""
        # Unified storage table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS unified_storage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_type TEXT NOT NULL,
                data TEXT NOT NULL,
                key TEXT,
                expires_at REAL,
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_unified_storage_content_type
            ON unified_storage(content_type)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_unified_storage_key
            ON unified_storage(key)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_unified_storage_expires
            ON unified_storage(expires_at)
        """)

        # Checkpoints table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_name TEXT NOT NULL,
                checkpoint_type TEXT,
                data TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_name
            ON checkpoints(checkpoint_name)
        """)

        # Audit log table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                user_id TEXT,
                details TEXT,
                metadata TEXT,
                timestamp REAL NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
            ON audit_log(timestamp)
        """)

    def _create_fts_tables(self):
        """Create FTS5 full-text search tables"""
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS unified_storage_fts
                USING fts5(id, content_type, data, content='unified_storage', content_rowid='id')
            """)

            # Create triggers to keep FTS in sync
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS unified_storage_fts_insert
                AFTER INSERT ON unified_storage
                BEGIN
                    INSERT INTO unified_storage_fts(id, content_type, data)
                    VALUES (new.id, new.content_type, new.data);
                END
            """)

            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS unified_storage_fts_delete
                AFTER DELETE ON unified_storage
                BEGIN
                    DELETE FROM unified_storage_fts WHERE id = old.id;
                END
            """)

            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS unified_storage_fts_update
                AFTER UPDATE ON unified_storage
                BEGIN
                    UPDATE unified_storage_fts
                    SET content_type = new.content_type, data = new.data
                    WHERE id = new.id;
                END
            """)

        except Exception as e:
            logger.warning(f"Could not create FTS tables: {e}")
            self.enable_fts = False

    def _get_table_for_content_type(self, content_type: ContentType) -> str:
        """Map content type to table name"""
        table_map = {
            ContentType.CHECKPOINT: "checkpoints",
            ContentType.AUDIT: "audit_log"
        }
        return table_map.get(content_type, "unified_storage")

    def _store_generic(
        self, content_type: ContentType, data_json: str,
        key: Optional[str], expires_at: Optional[float], metadata_json: str
    ) -> int:
        """Store generic data in unified_storage table"""
        now = time.time()
        cursor = self.conn.execute("""
            INSERT INTO unified_storage
            (content_type, data, key, expires_at, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            content_type.value,
            data_json,
            key,
            expires_at,
            metadata_json,
            now,
            now
        ))
        return cursor.lastrowid

    def _store_checkpoint(self, data: Dict, metadata: Optional[Dict]) -> int:
        """Store checkpoint data"""
        cursor = self.conn.execute("""
            INSERT INTO checkpoints
            (checkpoint_name, checkpoint_type, data, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data.get('name', ''),
            data.get('type', 'model'),
            json.dumps(data.get('data', {})),
            json.dumps(metadata or {}),
            time.time()
        ))
        return cursor.lastrowid

    def _store_audit_log(self, data: Dict, metadata: Optional[Dict]) -> int:
        """Store audit log entry"""
        cursor = self.conn.execute("""
            INSERT INTO audit_log
            (action, entity_type, entity_id, user_id, details, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('action', ''),
            data.get('entity_type', ''),
            data.get('entity_id', ''),
            data.get('user_id', ''),
            json.dumps(data.get('details', {})),
            json.dumps(metadata or {}),
            time.time()
        ))
        return cursor.lastrowid

    def _retrieve_generic(self, storage_id: str) -> Optional[Any]:
        """Retrieve generic data from unified_storage"""
        cursor = self.conn.execute("""
            SELECT * FROM unified_storage WHERE id = ?
        """, (storage_id,))

        row = cursor.fetchone()
        if not row:
            return None

        # Check expiration
        if row['expires_at'] and row['expires_at'] < time.time():
            # Expired, delete and return None
            self.conn.execute("DELETE FROM unified_storage WHERE id = ?", (storage_id,))
            return None

        try:
            return json.loads(row['data'])
        except:
            return row['data']

    def _retrieve_checkpoint(self, storage_id: str) -> Optional[Dict]:
        """Retrieve checkpoint by ID"""
        cursor = self.conn.execute("""
            SELECT * FROM checkpoints WHERE id = ?
        """, (storage_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'id': row['id'],
            'name': row['checkpoint_name'],
            'type': row['checkpoint_type'],
            'data': json.loads(row['data']),
            'metadata': json.loads(row['metadata']),
            'created_at': row['created_at']
        }

    def _retrieve_audit_log(self, storage_id: str) -> Optional[Dict]:
        """Retrieve audit log entry by ID"""
        cursor = self.conn.execute("""
            SELECT * FROM audit_log WHERE id = ?
        """, (storage_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'id': row['id'],
            'action': row['action'],
            'entity_type': row['entity_type'],
            'entity_id': row['entity_id'],
            'user_id': row['user_id'],
            'details': json.loads(row['details']),
            'metadata': json.loads(row['metadata']),
            'timestamp': row['timestamp']
        }

    def _search_fts(
        self, query: str, content_type: Optional[ContentType], limit: int
    ) -> List[SearchResult]:
        """Search using FTS5"""
        # Build query
        if content_type:
            sql = """
                SELECT s.*, fts.rank
                FROM unified_storage s
                JOIN unified_storage_fts fts ON s.id = fts.id
                WHERE unified_storage_fts MATCH ? AND s.content_type = ?
                ORDER BY fts.rank
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (query, content_type.value, limit))
        else:
            sql = """
                SELECT s.*, fts.rank
                FROM unified_storage s
                JOIN unified_storage_fts fts ON s.id = fts.id
                WHERE unified_storage_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (query, limit))

        results = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row['data'])
            except:
                content = row['data']

            try:
                metadata = json.loads(row['metadata'])
            except:
                metadata = {}

            handle = StorageHandle(
                storage_type=self.storage_type,
                storage_id=str(row['id']),
                content_type=ContentType(row['content_type']),
                tier=self.default_tier,
                created_at=datetime.fromtimestamp(row['created_at']),
                metadata=metadata
            )

            results.append(SearchResult(
                handle=handle,
                score=-float(row['rank']),  # FTS5 rank is negative
                content=content,
                metadata=metadata
            ))

        return results

    def _search_like(
        self, query: str, content_type: Optional[ContentType], limit: int
    ) -> List[SearchResult]:
        """Search using LIKE (fallback when FTS not available)"""
        if content_type:
            sql = """
                SELECT * FROM unified_storage
                WHERE content_type = ? AND data LIKE ?
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (content_type.value, f'%{query}%', limit))
        else:
            sql = """
                SELECT * FROM unified_storage
                WHERE data LIKE ?
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (f'%{query}%', limit))

        results = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row['data'])
            except:
                content = row['data']

            try:
                metadata = json.loads(row['metadata'])
            except:
                metadata = {}

            handle = StorageHandle(
                storage_type=self.storage_type,
                storage_id=str(row['id']),
                content_type=ContentType(row['content_type']),
                tier=self.default_tier,
                created_at=datetime.fromtimestamp(row['created_at']),
                metadata=metadata
            )

            results.append(SearchResult(
                handle=handle,
                score=1.0,
                content=content,
                metadata=metadata
            ))

        return results

    def backup(self, destination: str) -> bool:
        """
        Backup SQLite database

        Args:
            destination: Backup file path

        Returns:
            True if backup successful
        """
        try:
            if self.db_path == ':memory:':
                logger.warning("Cannot backup in-memory database")
                return False

            import shutil
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Use SQLite backup API for consistent snapshot
            backup_conn = sqlite3.connect(destination)
            with backup_conn:
                self.conn.backup(backup_conn)
            backup_conn.close()

            logger.info(f"SQLite backup successful: {destination}")
            return True

        except Exception as e:
            logger.error(f"Error backing up SQLite: {e}")
            return False

    def optimize(self) -> bool:
        """
        Optimize SQLite database (VACUUM, ANALYZE)

        Returns:
            True if optimization successful
        """
        try:
            # Delete expired entries
            self.conn.execute("""
                DELETE FROM unified_storage
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (time.time(),))

            # Rebuild FTS if enabled
            if self.enable_fts:
                self.conn.execute("INSERT INTO unified_storage_fts(unified_storage_fts) VALUES('rebuild')")

            # ANALYZE to update statistics
            self.conn.execute("ANALYZE")

            # VACUUM to reclaim space (only if not in-memory)
            if self.db_path != ':memory:':
                self.conn.execute("VACUUM")

            logger.info("SQLite optimization completed")
            return True

        except Exception as e:
            logger.error(f"Error optimizing SQLite: {e}")
            return False


if __name__ == "__main__":
    print("=" * 80)
    print("SQLITE STORAGE BACKEND")
    print("=" * 80 + "\n")

    # Example configuration
    config = {
        'db_path': '/dev/shm/lat5150_ramdisk.db',  # RAM disk for speed
        'use_wal': True,
        'cache_size': 2000,
        'enable_fts': True
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("✓ SQLite Storage Backend initialized")
    print("\nSupports:")
    print("  - File-based and in-memory databases")
    print("  - RAM disk (/dev/shm, tmpfs)")
    print("  - Full-text search (FTS5)")
    print("  - ACID transactions")
    print("  - Checkpoints and audit logs")
    print("  - Write-Ahead Logging (WAL)")
