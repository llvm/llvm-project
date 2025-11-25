#!/usr/bin/env python3
"""
Response Cache System with Redis

Caches AI responses to reduce latency and model load for repeated queries.

Features:
- Redis-based caching with TTL
- Cache key generation from query + model + parameters
- Hit rate tracking
- Automatic expiration
- PostgreSQL backup for persistence

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import os

# Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("WARNING: redis not installed. Caching disabled.")

# PostgreSQL for persistent cache
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


@dataclass
class CacheEntry:
    """Represents a cached response"""
    cache_key: str
    query_hash: str
    model: str
    response: str
    tokens: int
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int

    def to_dict(self) -> Dict:
        return {
            "cache_key": self.cache_key,
            "query_hash": self.query_hash,
            "model": self.model,
            "response": self.response,
            "tokens": self.tokens,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds
        }


class ResponseCache:
    """
    Response caching system with Redis and PostgreSQL backup

    Cache Strategy:
    - Primary: Redis (fast in-memory)
    - Backup: PostgreSQL (persistent storage)
    - TTL: Configurable expiration (default 1 hour)
    - LRU: Automatic eviction in Redis
    """

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 default_ttl: int = 3600,
                 use_postgres_backup: bool = True,
                 postgres_config: Optional[Dict] = None):
        """
        Initialize response cache

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default TTL in seconds (3600 = 1 hour)
            use_postgres_backup: Whether to use PostgreSQL for persistent cache
            postgres_config: PostgreSQL connection config
        """
        self.default_ttl = default_ttl
        self.use_postgres_backup = use_postgres_backup

        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                self.redis_enabled = True
                print(f"Redis cache connected: {redis_host}:{redis_port}/{redis_db}")
            except Exception as e:
                print(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
                self.redis_enabled = False
        else:
            self.redis_client = None
            self.redis_enabled = False

        # Initialize PostgreSQL backup
        self.postgres_conn = None
        if use_postgres_backup and POSTGRES_AVAILABLE and postgres_config:
            try:
                self.postgres_conn = psycopg2.connect(**postgres_config)
                print("PostgreSQL cache backup connected")
            except Exception as e:
                print(f"PostgreSQL connection failed: {e}. No persistent cache backup.")

        # Cache stats
        self.hits = 0
        self.misses = 0

    def _generate_cache_key(self,
                           query: str,
                           model: str,
                           temperature: Optional[float] = None,
                           max_tokens: Optional[int] = None,
                           **kwargs) -> str:
        """
        Generate cache key from query and parameters

        Args:
            query: User query
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            **kwargs: Additional parameters

        Returns:
            SHA256 hash as cache key
        """
        # Normalize parameters
        cache_params = {
            "query": query.strip().lower(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Create deterministic string
        cache_str = json.dumps(cache_params, sort_keys=True)

        # Generate hash
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(self,
           query: str,
           model: str,
           temperature: Optional[float] = None,
           max_tokens: Optional[int] = None,
           **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available

        Returns:
            Dict with response and metadata, or None if not cached
        """
        cache_key = self._generate_cache_key(query, model, temperature, max_tokens, **kwargs)

        # Try Redis first
        if self.redis_enabled:
            cached_json = self.redis_client.get(f"cache:{cache_key}")
            if cached_json:
                self.hits += 1
                cached_data = json.loads(cached_json)
                cached_data['cache_hit'] = True
                cached_data['cache_source'] = 'redis'

                # Update access stats
                self._update_access_stats(cache_key)

                return cached_data

        # Try PostgreSQL backup
        if self.use_postgres_backup and self.postgres_conn:
            try:
                with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT response, tokens, created_at, access_count
                        FROM response_cache
                        WHERE cache_key = %s AND expires_at > NOW()
                    """, (cache_key,))

                    result = cur.fetchone()
                    if result:
                        self.hits += 1

                        # Update access stats
                        cur.execute("""
                            UPDATE response_cache
                            SET accessed_at = NOW(), access_count = access_count + 1
                            WHERE cache_key = %s
                        """, (cache_key,))
                        self.postgres_conn.commit()

                        return {
                            'response': result['response'],
                            'tokens': result['tokens'],
                            'cache_hit': True,
                            'cache_source': 'postgres',
                            'created_at': result['created_at'].isoformat()
                        }
            except Exception as e:
                print(f"PostgreSQL cache read error: {e}")

        # Cache miss
        self.misses += 1
        return None

    def set(self,
           query: str,
           model: str,
           response: str,
           tokens: int,
           temperature: Optional[float] = None,
           max_tokens: Optional[int] = None,
           ttl: Optional[int] = None,
           **kwargs):
        """
        Cache a response

        Args:
            query: User query
            model: Model name
            response: Model response
            tokens: Token count
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            ttl: TTL in seconds (None = use default)
            **kwargs: Additional parameters
        """
        cache_key = self._generate_cache_key(query, model, temperature, max_tokens, **kwargs)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        ttl = ttl or self.default_ttl

        cache_data = {
            'response': response,
            'tokens': tokens,
            'model': model,
            'query_hash': query_hash,
            'created_at': datetime.now().isoformat(),
            'ttl': ttl
        }

        # Store in Redis
        if self.redis_enabled:
            try:
                self.redis_client.setex(
                    f"cache:{cache_key}",
                    ttl,
                    json.dumps(cache_data)
                )
            except Exception as e:
                print(f"Redis cache write error: {e}")

        # Store in PostgreSQL backup
        if self.use_postgres_backup and self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    expires_at = datetime.now() + timedelta(seconds=ttl)

                    cur.execute("""
                        INSERT INTO response_cache
                        (cache_key, query_hash, model, response, tokens, ttl_seconds, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (cache_key) DO UPDATE
                        SET response = EXCLUDED.response,
                            tokens = EXCLUDED.tokens,
                            accessed_at = NOW(),
                            access_count = response_cache.access_count + 1,
                            expires_at = EXCLUDED.expires_at
                    """, (cache_key, query_hash, model, response, tokens, ttl, expires_at))

                    self.postgres_conn.commit()
            except Exception as e:
                print(f"PostgreSQL cache write error: {e}")

    def _update_access_stats(self, cache_key: str):
        """Update access statistics for a cache entry"""
        if self.use_postgres_backup and self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("""
                        UPDATE response_cache
                        SET accessed_at = NOW(), access_count = access_count + 1
                        WHERE cache_key = %s
                    """, (cache_key,))
                    self.postgres_conn.commit()
            except Exception as e:
                print(f"Failed to update access stats: {e}")

    def invalidate(self, query: str, model: str, **kwargs):
        """Invalidate a specific cache entry"""
        cache_key = self._generate_cache_key(query, model, **kwargs)

        # Remove from Redis
        if self.redis_enabled:
            self.redis_client.delete(f"cache:{cache_key}")

        # Remove from PostgreSQL
        if self.use_postgres_backup and self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM response_cache WHERE cache_key = %s
                    """, (cache_key,))
                    self.postgres_conn.commit()
            except Exception as e:
                print(f"Failed to invalidate from PostgreSQL: {e}")

    def clear_expired(self) -> int:
        """
        Clear expired cache entries from PostgreSQL

        Returns:
            Number of entries deleted
        """
        if not (self.use_postgres_backup and self.postgres_conn):
            return 0

        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute("SELECT cleanup_expired_cache()")
                deleted_count = cur.fetchone()[0]
                self.postgres_conn.commit()
                return deleted_count
        except Exception as e:
            print(f"Failed to clear expired cache: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        stats = {
            'redis_enabled': self.redis_enabled,
            'postgres_backup': self.use_postgres_backup and self.postgres_conn is not None,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'hit_rate_percent': f"{hit_rate * 100:.1f}%"
        }

        # Redis stats
        if self.redis_enabled:
            try:
                info = self.redis_client.info('stats')
                stats['redis_keys'] = self.redis_client.dbsize()
                stats['redis_memory'] = self.redis_client.info('memory').get('used_memory_human', 'N/A')
            except Exception as e:
                stats['redis_error'] = str(e)

        # PostgreSQL stats
        if self.use_postgres_backup and self.postgres_conn:
            try:
                with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_entries,
                            SUM(access_count) as total_accesses,
                            AVG(access_count) as avg_accesses
                        FROM response_cache
                        WHERE expires_at > NOW()
                    """)
                    result = cur.fetchone()
                    stats['postgres_entries'] = result['total_entries']
                    stats['postgres_total_accesses'] = result['total_accesses']
                    stats['postgres_avg_accesses'] = float(result['avg_accesses'] or 0)
            except Exception as e:
                stats['postgres_error'] = str(e)

        return stats

    def close(self):
        """Close connections"""
        if self.postgres_conn:
            self.postgres_conn.close()
        if self.redis_client:
            self.redis_client.close()


# Example usage and testing
if __name__ == "__main__":
    print("Response Cache Test")
    print("=" * 60)

    # Initialize cache (will work with just Redis, PostgreSQL optional)
    cache = ResponseCache()

    # Cache a response
    query = "What is the context window size?"
    model = "uncensored_code"
    response = "The context window is 8192 tokens for all models."

    print(f"\nCaching response for: '{query}'")
    cache.set(query, model, response, tokens=15)

    # Retrieve cached response
    print(f"\nRetrieving cached response...")
    cached = cache.get(query, model)

    if cached:
        print(f"✓ Cache hit! Source: {cached.get('cache_source', 'unknown')}")
        print(f"  Response: {cached['response']}")
    else:
        print("✗ Cache miss")

    # Get stats
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Redis: {stats['redis_enabled']}")
    print(f"  PostgreSQL: {stats['postgres_backup']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate_percent']}")

    # Close
    cache.close()
