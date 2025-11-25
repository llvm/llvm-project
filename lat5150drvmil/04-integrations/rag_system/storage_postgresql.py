#!/usr/bin/env python3
"""
PostgreSQL Storage Backend Adapter

Provides unified interface for PostgreSQL databases in the LAT5150DRVMIL AI engine.
Handles:
- Conversations and messages
- Knowledge graph data
- Long-term memory blocks
- RAG embeddings and metadata
- Transaction support with ACID guarantees
"""

import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import ThreadedConnectionPool
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


class PostgreSQLStorageBackend(AbstractStorageBackend):
    """
    PostgreSQL storage adapter for AI engine data

    Supports:
    - Conversations, messages, documents
    - Metadata and structured data
    - Full-text search via PostgreSQL
    - Transaction support
    - Connection pooling
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL backend

        Args:
            config: Configuration dictionary with:
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Username
                - password: Password
                - min_connections: Min pool size (default: 2)
                - max_connections: Max pool size (default: 10)
                - schema: Schema name (default: public)
        """
        super().__init__(config)
        self.storage_type = StorageType.POSTGRESQL

        # Connection configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 5432)
        self.database = config.get('database', 'ai_engine')
        self.user = config.get('user', 'postgres')
        self.password = config.get('password', '')
        self.schema = config.get('schema', 'public')

        # Connection pooling
        self.min_connections = config.get('min_connections', 2)
        self.max_connections = config.get('max_connections', 10)
        self.pool = None

        # Performance tracking
        self._query_times = []
        self._max_query_history = 1000

    def connect(self) -> bool:
        """
        Establish connection pool to PostgreSQL

        Returns:
            True if connection successful
        """
        try:
            self.pool = ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )

            # Test connection
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    logger.info(f"Connected to PostgreSQL: {version}")
            finally:
                self.pool.putconn(conn)

            # Ensure schema exists
            self._ensure_schema()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Close all connections in pool

        Returns:
            True if disconnection successful
        """
        try:
            if self.pool:
                self.pool.closeall()
                logger.info("PostgreSQL connection pool closed")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {e}")
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
        Store data in PostgreSQL

        Args:
            data: Data to store (dict, str, or JSON-serializable)
            content_type: Type of content
            key: Optional key (auto-generated if not provided)
            ttl: Time-to-live in seconds (for expiring data)
            metadata: Additional metadata

        Returns:
            StorageHandle for retrieving data
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Determine table and handle serialization
                table_name = self._get_table_for_content_type(content_type)

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

                # Insert data
                if content_type == ContentType.CONVERSATION:
                    storage_id = self._store_conversation(cur, data, metadata)
                elif content_type == ContentType.MESSAGE:
                    storage_id = self._store_message(cur, data, metadata)
                elif content_type == ContentType.DOCUMENT:
                    storage_id = self._store_document(cur, data, metadata)
                elif content_type == ContentType.MEMORY:
                    storage_id = self._store_memory_block(cur, data, metadata)
                elif content_type == ContentType.METADATA:
                    storage_id = self._store_metadata(cur, data, key, metadata)
                else:
                    # Generic storage in unified_storage table
                    storage_id = self._store_generic(
                        cur, content_type, data_json, key, expires_at, metadata_json
                    )

                conn.commit()
                self._stats['operations'] += 1
                self._stats['bytes_written'] += len(data_json.encode())

                return StorageHandle(
                    storage_type=self.storage_type,
                    storage_id=storage_id,
                    content_type=content_type,
                    tier=StorageTier.WARM,
                    created_at=datetime.now(),
                    metadata=metadata or {}
                )

        except Exception as e:
            conn.rollback()
            self._stats['errors'] += 1
            logger.error(f"Error storing data in PostgreSQL: {e}")
            raise
        finally:
            self.pool.putconn(conn)

    def retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """
        Retrieve data by handle

        Args:
            handle: Storage handle from store()

        Returns:
            Retrieved data or None if not found
        """
        conn = self.pool.getconn()
        try:
            start_time = time.time()

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if handle.content_type == ContentType.CONVERSATION:
                    data = self._retrieve_conversation(cur, handle.storage_id)
                elif handle.content_type == ContentType.MESSAGE:
                    data = self._retrieve_message(cur, handle.storage_id)
                elif handle.content_type == ContentType.DOCUMENT:
                    data = self._retrieve_document(cur, handle.storage_id)
                elif handle.content_type == ContentType.MEMORY:
                    data = self._retrieve_memory_block(cur, handle.storage_id)
                elif handle.content_type == ContentType.METADATA:
                    data = self._retrieve_metadata(cur, handle.storage_id)
                else:
                    data = self._retrieve_generic(cur, handle.storage_id)

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
            logger.error(f"Error retrieving data from PostgreSQL: {e}")
            return None
        finally:
            self.pool.putconn(conn)

    def delete(self, handle: StorageHandle) -> bool:
        """
        Delete data by handle

        Args:
            handle: Storage handle

        Returns:
            True if deletion successful
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                table_name = self._get_table_for_content_type(handle.content_type)

                cur.execute(
                    f"DELETE FROM {self.schema}.{table_name} WHERE id = %s",
                    (handle.storage_id,)
                )

                deleted = cur.rowcount > 0
                conn.commit()
                self._stats['operations'] += 1

                return deleted

        except Exception as e:
            conn.rollback()
            self._stats['errors'] += 1
            logger.error(f"Error deleting data from PostgreSQL: {e}")
            return False
        finally:
            self.pool.putconn(conn)

    def search(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search for data using PostgreSQL full-text search

        Args:
            query: Search query text
            content_type: Filter by content type
            filters: Additional filters
            limit: Maximum results

        Returns:
            List of search results
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build search query
                if content_type:
                    table_name = self._get_table_for_content_type(content_type)
                    search_results = self._search_table(
                        cur, table_name, query, filters, limit, content_type
                    )
                else:
                    # Search across all tables
                    search_results = []
                    for ct in ContentType:
                        table_name = self._get_table_for_content_type(ct)
                        results = self._search_table(
                            cur, table_name, query, filters, limit, ct
                        )
                        search_results.extend(results)

                    # Sort by score and limit
                    search_results.sort(key=lambda x: x.score, reverse=True)
                    search_results = search_results[:limit]

                self._stats['operations'] += 1
                return search_results

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error searching PostgreSQL: {e}")
            return []
        finally:
            self.pool.putconn(conn)

    def get_stats(self) -> StorageStats:
        """
        Get PostgreSQL storage statistics

        Returns:
            StorageStats object
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get database size
                cur.execute(f"""
                    SELECT pg_database_size('{self.database}') as size_bytes
                """)
                size_result = cur.fetchone()
                total_size = size_result['size_bytes'] if size_result else 0

                # Count total items across main tables
                total_items = 0
                for content_type in ContentType:
                    table_name = self._get_table_for_content_type(content_type)
                    try:
                        cur.execute(f"""
                            SELECT COUNT(*) as count
                            FROM {self.schema}.{table_name}
                        """)
                        count_result = cur.fetchone()
                        if count_result:
                            total_items += count_result['count']
                    except:
                        pass  # Table might not exist

                # Calculate average access time
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
                        'pool_size': self.max_connections,
                        'operations': self._stats['operations'],
                        'bytes_written': self._stats['bytes_written'],
                        'bytes_read': self._stats['bytes_read'],
                        'errors': self._stats['errors']
                    }
                )

        except Exception as e:
            logger.error(f"Error getting PostgreSQL stats: {e}")
            return StorageStats(
                storage_type=self.storage_type,
                total_items=0,
                total_size_bytes=0,
                avg_access_time_ms=0,
                health_status="error"
            )
        finally:
            self.pool.putconn(conn)

    def health_check(self) -> Tuple[bool, str]:
        """
        Check PostgreSQL health

        Returns:
            (is_healthy, status_message)
        """
        try:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    if result and result[0] == 1:
                        return (True, "PostgreSQL connection healthy")
                    else:
                        return (False, "Unexpected health check result")
            finally:
                self.pool.putconn(conn)
        except Exception as e:
            return (False, f"PostgreSQL health check failed: {e}")

    # Internal helper methods

    def _ensure_schema(self):
        """Ensure schema and unified storage table exist"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Create schema if needed
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

                # Create unified storage table for generic content
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.unified_storage (
                        id SERIAL PRIMARY KEY,
                        content_type VARCHAR(50) NOT NULL,
                        data JSONB NOT NULL,
                        key VARCHAR(255),
                        expires_at TIMESTAMP,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create indices
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_unified_storage_content_type
                    ON {self.schema}.unified_storage(content_type)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_unified_storage_key
                    ON {self.schema}.unified_storage(key)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_unified_storage_expires
                    ON {self.schema}.unified_storage(expires_at)
                """)

                conn.commit()
        finally:
            self.pool.putconn(conn)

    def _get_table_for_content_type(self, content_type: ContentType) -> str:
        """Map content type to table name"""
        table_map = {
            ContentType.CONVERSATION: "conversations",
            ContentType.MESSAGE: "messages",
            ContentType.DOCUMENT: "documents",
            ContentType.MEMORY: "memory_blocks",
            ContentType.METADATA: "metadata_store",
            ContentType.CHECKPOINT: "checkpoints",
            ContentType.AUDIT: "audit_log"
        }
        return table_map.get(content_type, "unified_storage")

    def _store_conversation(self, cur, data: Dict, metadata: Optional[Dict]) -> str:
        """Store conversation data"""
        cur.execute(f"""
            INSERT INTO {self.schema}.conversations
            (title, user_id, metadata, created_at)
            VALUES (%s, %s, %s, NOW())
            RETURNING id
        """, (
            data.get('title', ''),
            data.get('user_id', ''),
            json.dumps(metadata or {})
        ))
        return str(cur.fetchone()[0])

    def _store_message(self, cur, data: Dict, metadata: Optional[Dict]) -> str:
        """Store message data"""
        cur.execute(f"""
            INSERT INTO {self.schema}.messages
            (conversation_id, role, content, metadata, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
        """, (
            data.get('conversation_id'),
            data.get('role', 'user'),
            data.get('content', ''),
            json.dumps(metadata or {})
        ))
        return str(cur.fetchone()[0])

    def _store_document(self, cur, data: Dict, metadata: Optional[Dict]) -> str:
        """Store document data"""
        cur.execute(f"""
            INSERT INTO {self.schema}.documents
            (title, content, doc_type, metadata, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
        """, (
            data.get('title', ''),
            data.get('content', ''),
            data.get('doc_type', 'text'),
            json.dumps(metadata or {})
        ))
        return str(cur.fetchone()[0])

    def _store_memory_block(self, cur, data: Dict, metadata: Optional[Dict]) -> str:
        """Store memory block data"""
        cur.execute(f"""
            INSERT INTO {self.schema}.memory_blocks
            (content, memory_type, importance, metadata, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
        """, (
            data.get('content', ''),
            data.get('memory_type', 'episodic'),
            data.get('importance', 0.5),
            json.dumps(metadata or {})
        ))
        return str(cur.fetchone()[0])

    def _store_metadata(self, cur, data: Dict, key: str, metadata: Optional[Dict]) -> str:
        """Store metadata"""
        cur.execute(f"""
            INSERT INTO {self.schema}.metadata_store
            (key, value, metadata, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id
        """, (
            key,
            json.dumps(data),
            json.dumps(metadata or {})
        ))
        return str(cur.fetchone()[0])

    def _store_generic(
        self, cur, content_type: ContentType, data_json: str,
        key: Optional[str], expires_at: Optional[float], metadata_json: str
    ) -> str:
        """Store generic data in unified_storage table"""
        cur.execute(f"""
            INSERT INTO {self.schema}.unified_storage
            (content_type, data, key, expires_at, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            content_type.value,
            data_json,
            key,
            datetime.fromtimestamp(expires_at) if expires_at else None,
            metadata_json
        ))
        return str(cur.fetchone()[0])

    def _retrieve_conversation(self, cur, storage_id: str) -> Optional[Dict]:
        """Retrieve conversation by ID"""
        cur.execute(f"""
            SELECT * FROM {self.schema}.conversations WHERE id = %s
        """, (storage_id,))
        return dict(cur.fetchone()) if cur.rowcount > 0 else None

    def _retrieve_message(self, cur, storage_id: str) -> Optional[Dict]:
        """Retrieve message by ID"""
        cur.execute(f"""
            SELECT * FROM {self.schema}.messages WHERE id = %s
        """, (storage_id,))
        return dict(cur.fetchone()) if cur.rowcount > 0 else None

    def _retrieve_document(self, cur, storage_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        cur.execute(f"""
            SELECT * FROM {self.schema}.documents WHERE id = %s
        """, (storage_id,))
        return dict(cur.fetchone()) if cur.rowcount > 0 else None

    def _retrieve_memory_block(self, cur, storage_id: str) -> Optional[Dict]:
        """Retrieve memory block by ID"""
        cur.execute(f"""
            SELECT * FROM {self.schema}.memory_blocks WHERE id = %s
        """, (storage_id,))
        return dict(cur.fetchone()) if cur.rowcount > 0 else None

    def _retrieve_metadata(self, cur, storage_id: str) -> Optional[Dict]:
        """Retrieve metadata by ID"""
        cur.execute(f"""
            SELECT * FROM {self.schema}.metadata_store WHERE id = %s
        """, (storage_id,))
        return dict(cur.fetchone()) if cur.rowcount > 0 else None

    def _retrieve_generic(self, cur, storage_id: str) -> Optional[Any]:
        """Retrieve generic data from unified_storage"""
        cur.execute(f"""
            SELECT data FROM {self.schema}.unified_storage WHERE id = %s
        """, (storage_id,))
        result = cur.fetchone()
        if result:
            return json.loads(result['data']) if isinstance(result['data'], str) else result['data']
        return None

    def _search_table(
        self, cur, table_name: str, query: str,
        filters: Optional[Dict], limit: int, content_type: ContentType
    ) -> List[SearchResult]:
        """Search a specific table with full-text search"""
        try:
            # Basic full-text search on common fields
            cur.execute(f"""
                SELECT *,
                       ts_rank(to_tsvector('english', COALESCE(content, '') || ' ' || COALESCE(title, '')),
                               plainto_tsquery('english', %s)) as rank
                FROM {self.schema}.{table_name}
                WHERE to_tsvector('english', COALESCE(content, '') || ' ' || COALESCE(title, '')) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """, (query, query, limit))

            results = []
            for row in cur.fetchall():
                row_dict = dict(row)
                score = row_dict.pop('rank', 0.0)

                handle = StorageHandle(
                    storage_type=self.storage_type,
                    storage_id=str(row_dict['id']),
                    content_type=content_type,
                    tier=StorageTier.WARM,
                    created_at=row_dict.get('created_at', datetime.now()),
                    metadata=row_dict.get('metadata', {})
                )

                results.append(SearchResult(
                    handle=handle,
                    score=float(score),
                    content=row_dict,
                    metadata=row_dict.get('metadata', {})
                ))

            return results

        except Exception as e:
            logger.warning(f"Error searching table {table_name}: {e}")
            return []

    def backup(self, destination: str) -> bool:
        """
        Backup PostgreSQL database using pg_dump

        Args:
            destination: Backup file path

        Returns:
            True if backup successful
        """
        import subprocess
        try:
            cmd = [
                'pg_dump',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-F', 'c',  # Custom format
                '-f', destination
            ]

            env = {'PGPASSWORD': self.password}
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"PostgreSQL backup successful: {destination}")
                return True
            else:
                logger.error(f"PostgreSQL backup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error backing up PostgreSQL: {e}")
            return False

    def restore(self, source: str) -> bool:
        """
        Restore PostgreSQL database from backup

        Args:
            source: Backup file path

        Returns:
            True if restore successful
        """
        import subprocess
        try:
            cmd = [
                'pg_restore',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-c',  # Clean before restore
                source
            ]

            env = {'PGPASSWORD': self.password}
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"PostgreSQL restore successful from: {source}")
                return True
            else:
                logger.error(f"PostgreSQL restore failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error restoring PostgreSQL: {e}")
            return False

    def optimize(self) -> bool:
        """
        Optimize PostgreSQL (VACUUM ANALYZE)

        Returns:
            True if optimization successful
        """
        conn = self.pool.getconn()
        try:
            old_isolation = conn.isolation_level
            conn.set_isolation_level(0)  # AUTOCOMMIT for VACUUM

            with conn.cursor() as cur:
                cur.execute("VACUUM ANALYZE")
                logger.info("PostgreSQL optimization (VACUUM ANALYZE) completed")

            conn.set_isolation_level(old_isolation)
            return True

        except Exception as e:
            logger.error(f"Error optimizing PostgreSQL: {e}")
            return False
        finally:
            self.pool.putconn(conn)


if __name__ == "__main__":
    print("=" * 80)
    print("POSTGRESQL STORAGE BACKEND")
    print("=" * 80 + "\n")

    # Example configuration
    config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'ai_engine',
        'user': 'postgres',
        'password': 'password',
        'min_connections': 2,
        'max_connections': 10
    }

    print("Configuration:")
    for key, value in config.items():
        if key != 'password':
            print(f"  {key}: {value}")
    print()

    print("✓ PostgreSQL Storage Backend initialized")
    print("\nSupports:")
    print("  - Conversations, messages, documents")
    print("  - Memory blocks and metadata")
    print("  - Full-text search")
    print("  - Connection pooling")
    print("  - Backup/restore with pg_dump")
    print("  - VACUUM optimization")
