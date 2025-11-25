# Unified Storage System for LAT5150DRVMIL AI Engine

**Version:** 1.0.0
**Status:** Production Ready
**Created:** 2025-11-13

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Storage Backends](#storage-backends)
4. [Storage Orchestrator](#storage-orchestrator)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The **Unified Storage System** provides a comprehensive, production-ready abstraction layer for all storage operations in the LAT5150DRVMIL AI engine. It intelligently manages multiple storage backends, automatic caching, replication, and lifecycle management.

### Key Features

✅ **Unified Interface** - Single API for all storage operations
✅ **Intelligent Routing** - Automatic backend selection based on content type
✅ **Multi-Backend Support** - PostgreSQL, Redis, Qdrant, SQLite
✅ **Automatic Caching** - Redis-based caching with configurable TTL
✅ **Vector Search** - Qdrant integration for embeddings (Jina v3, BGE)
✅ **Health Monitoring** - Continuous health checks and failover
✅ **Backup/Recovery** - Coordinated backup across all backends
✅ **Storage Tiers** - HOT/WARM/COLD/FROZEN lifecycle management
✅ **Transaction Support** - ACID guarantees where applicable

### Supported Storage Backends

| Backend | Use Case | Tier | Features |
|---------|----------|------|----------|
| **PostgreSQL** | Conversations, messages, documents | WARM | Full-text search, ACID, replication |
| **Redis** | Caching, sessions, counters | HOT | Sub-millisecond access, pub/sub, locks |
| **Qdrant** | Vector embeddings, semantic search | WARM | HNSW, quantization, multi-vector |
| **SQLite** | Checkpoints, audit logs, RAM disk | HOT/WARM | FTS5, WAL, in-memory |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│           (RAG System, AI Engine, User Code)                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│               Storage Orchestrator                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Intelligent Routing                               │   │
│  │  • Policy Management                                 │   │
│  │  • Caching Strategy                                  │   │
│  │  • Health Monitoring                                 │   │
│  │  • Backup Coordination                               │   │
│  └─────────────────────────────────────────────────────┘   │
└──────┬─────────┬──────────┬──────────┬────────────────────┘
       │         │          │          │
       ▼         ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌────────┐
│PostgreSQL│ │ Redis  │ │ Qdrant  │ │ SQLite │
│         │ │        │ │         │ │        │
│ Tables  │ │ KV     │ │ Vectors │ │ Files  │
│ FTS     │ │ Pub/Sub│ │ HNSW    │ │ FTS5   │
│ ACID    │ │ Locks  │ │ Quant   │ │ WAL    │
└─────────┘ └────────┘ └─────────┘ └────────┘
```

### Component Structure

```
04-integrations/rag_system/
├── storage_abstraction.py      # Base interfaces and types
├── storage_postgresql.py       # PostgreSQL adapter
├── storage_redis.py            # Redis adapter
├── storage_qdrant.py           # Qdrant vector adapter
├── storage_sqlite.py           # SQLite adapter
├── storage_orchestrator.py     # Main orchestration layer
└── UNIFIED_STORAGE_SYSTEM.md   # This documentation
```

---

## Storage Backends

### PostgreSQL Storage Backend

**Use Cases:** Conversations, messages, documents, long-term memory

```python
from storage_postgresql import PostgreSQLStorageBackend
from storage_abstraction import ContentType

# Configure
config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_engine',
    'user': 'postgres',
    'password': 'password',
    'min_connections': 2,
    'max_connections': 10
}

# Initialize
pg = PostgreSQLStorageBackend(config)
pg.connect()

# Store conversation
handle = pg.store(
    data={
        'title': 'Cyber Investigation #42',
        'user_id': 'analyst_007',
        'content': 'Analysis of ransomware attack...'
    },
    content_type=ContentType.CONVERSATION,
    metadata={'priority': 'high'}
)

# Retrieve
conversation = pg.retrieve(handle)
print(conversation['title'])  # "Cyber Investigation #42"

# Search with full-text
results = pg.search(
    query="ransomware",
    content_type=ContentType.CONVERSATION,
    limit=10
)
```

**Features:**
- Connection pooling (2-10 connections)
- Full-text search via PostgreSQL
- ACID transactions
- Automatic schema creation
- Backup via `pg_dump`
- VACUUM optimization

---

### Redis Cache Backend

**Use Cases:** High-speed caching, session storage, distributed locks

```python
from storage_redis import RedisStorageBackend
from storage_abstraction import ContentType

# Configure
config = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'max_connections': 50,
    'key_prefix': 'lat5150:'
}

# Initialize
redis = RedisStorageBackend(config)
redis.connect()

# Store with TTL
handle = redis.store(
    data={'query_result': [...]},
    content_type=ContentType.CACHE,
    key='search:ransomware:results',
    ttl=3600  # 1 hour
)

# Cache-specific methods
redis.cache_set('user:session:123', {'user_id': 123}, ttl=1800)
session = redis.cache_get('user:session:123')

# Distributed lock
lock = redis.acquire_lock('process:analysis', timeout=30)
if lock:
    # Critical section
    redis.release_lock(lock)

# Pub/sub
redis.publish('alerts', {'type': 'threat_detected', 'severity': 'high'})

# Counters
redis.increment('api:requests:count')

# Get hit rate
hit_rate = redis.get_cache_hit_rate()
print(f"Cache hit rate: {hit_rate * 100:.1f}%")
```

**Features:**
- Sub-millisecond access times
- Automatic expiration (TTL)
- Distributed locking
- Pub/sub messaging
- Atomic counters
- Pattern-based search
- Cache hit/miss tracking

---

### Qdrant Vector Backend

**Use Cases:** Vector embeddings, semantic search, RAG retrieval

```python
from storage_qdrant import QdrantStorageBackend
from storage_abstraction import ContentType

# Configure for Jina v3 embeddings
config = {
    'host': 'localhost',
    'port': 6333,
    'collection_name': 'jina_v3_embeddings',
    'vector_size': 1024,  # Jina v3 Matryoshka 1024D
    'distance': 'cosine',
    'use_quantization': True,  # INT8 quantization
    'hnsw_m': 32,
    'hnsw_ef_construct': 200
}

# Initialize
qdrant = QdrantStorageBackend(config)
qdrant.connect()

# Store embedding
handle = qdrant.store_embedding(
    text="Analysis of malware behavior in network traffic...",
    embedding=[0.123, -0.456, ...],  # 1024D vector
    content_type=ContentType.EMBEDDING,
    metadata={'doc_type': 'report', 'date': '2024-01-15'}
)

# Vector search
results = qdrant.vector_search(
    query_embedding=[0.789, -0.012, ...],
    content_type=ContentType.EMBEDDING,
    filters={'doc_type': 'report'},
    top_k=10
)

for result in results:
    print(f"Score: {result.score:.3f} - {result.content['text'][:100]}")

# Batch store
handles = qdrant.batch_store_embeddings(
    texts=['doc1', 'doc2', 'doc3'],
    embeddings=[emb1, emb2, emb3]
)

# Hybrid search (vector + text)
results = qdrant.hybrid_search(
    query_text="malware analysis",
    query_embedding=query_emb,
    alpha=0.7,  # 70% vector, 30% text
    top_k=10
)
```

**Features:**
- High-dimensional vectors (up to 2084D)
- HNSW indexing (m=32, ef=200)
- Scalar quantization (INT8, 4x compression)
- Multi-vector storage (ColBERT-style)
- Payload filtering
- Batch operations
- Hybrid search

---

### SQLite Storage Backend

**Use Cases:** Checkpoints, audit logs, RAM disk for high-speed access

```python
from storage_sqlite import SQLiteStorageBackend
from storage_abstraction import ContentType

# Configure for RAM disk
config = {
    'db_path': '/dev/shm/lat5150_ramdisk.db',  # RAM disk
    'use_wal': True,
    'cache_size': 2000,  # 2MB page cache
    'enable_fts': True   # Full-text search
}

# Or in-memory
config_memory = {'db_path': ':memory:'}

# Initialize
sqlite = SQLiteStorageBackend(config)
sqlite.connect()

# Store checkpoint
handle = sqlite.store(
    data={
        'name': 'model_epoch_42',
        'type': 'model',
        'data': {'weights': [...], 'optimizer': {...}}
    },
    content_type=ContentType.CHECKPOINT
)

# Store audit log
handle = sqlite.store(
    data={
        'action': 'document_accessed',
        'entity_type': 'report',
        'entity_id': 'report_123',
        'user_id': 'analyst_007',
        'details': {'ip': '10.0.1.50', 'timestamp': '2024-01-15T10:30:00'}
    },
    content_type=ContentType.AUDIT
)

# Full-text search (FTS5)
results = sqlite.search(
    query="model checkpoint",
    content_type=ContentType.CHECKPOINT,
    limit=10
)

# Optimization
sqlite.optimize()  # VACUUM, ANALYZE, rebuild FTS
```

**Features:**
- File-based or in-memory
- RAM disk support (/dev/shm, tmpfs)
- Full-text search (FTS5)
- Write-Ahead Logging (WAL)
- ACID transactions
- Automatic expiration
- Backup with consistent snapshots

---

## Storage Orchestrator

The **StorageOrchestrator** provides intelligent, unified management across all backends.

### Initialization

```python
from storage_orchestrator import StorageOrchestrator, OrchestratorConfig, StoragePolicy
from storage_abstraction import ContentType, StorageType

# Configure
config = OrchestratorConfig(
    postgresql_config={
        'host': 'localhost',
        'port': 5432,
        'database': 'ai_engine',
        'user': 'postgres',
        'password': 'password'
    },
    redis_config={
        'host': 'localhost',
        'port': 6379,
        'db': 0
    },
    qdrant_config={
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'embeddings',
        'vector_size': 1024,
        'use_quantization': True
    },
    sqlite_config={
        'db_path': '/dev/shm/lat5150_ramdisk.db',
        'use_wal': True,
        'enable_fts': True
    },
    enable_caching=True,
    enable_health_checks=True,
    health_check_interval=60,
    enable_auto_optimization=True,
    optimization_interval=3600
)

# Initialize
orchestrator = StorageOrchestrator(config)
orchestrator.initialize()

# Or use default configuration
from storage_orchestrator import create_default_orchestrator
orchestrator = create_default_orchestrator()
```

### Storage Policies

The orchestrator uses **storage policies** to determine backend routing:

```python
# Default policies
policies = {
    ContentType.CONVERSATION: StoragePolicy(
        primary_backend=StorageType.POSTGRESQL,
        cache_backend=StorageType.REDIS,
        cache_ttl=3600  # 1 hour
    ),

    ContentType.MESSAGE: StoragePolicy(
        primary_backend=StorageType.POSTGRESQL,
        cache_backend=StorageType.REDIS,
        cache_ttl=1800  # 30 minutes
    ),

    ContentType.EMBEDDING: StoragePolicy(
        primary_backend=StorageType.VECTOR,  # Qdrant
        cache_backend=StorageType.REDIS,
        cache_ttl=7200  # 2 hours
    ),

    ContentType.CHECKPOINT: StoragePolicy(
        primary_backend=StorageType.SQLITE
    ),

    ContentType.METADATA: StoragePolicy(
        primary_backend=StorageType.REDIS,
        enable_replication=True,
        replica_backend=StorageType.POSTGRESQL
    )
}

# Add custom policy
config.policies[ContentType.DOCUMENT] = StoragePolicy(
    primary_backend=StorageType.POSTGRESQL,
    cache_backend=StorageType.REDIS,
    cache_ttl=1800,
    enable_replication=True,
    replica_backend=StorageType.SQLITE
)
```

---

## Quick Start

### 1. Basic Storage Operations

```python
from storage_orchestrator import create_default_orchestrator
from storage_abstraction import ContentType

# Initialize
orchestrator = create_default_orchestrator()

# Store data (automatically routes to correct backend)
handle = orchestrator.store(
    data={'title': 'Investigation Report', 'content': '...'},
    content_type=ContentType.DOCUMENT,
    metadata={'priority': 'high'}
)

# Retrieve (with automatic caching)
document = orchestrator.retrieve(handle)

# Search across backends
results = orchestrator.search(
    query="ransomware attack",
    content_type=ContentType.DOCUMENT,
    limit=10
)

# Delete
orchestrator.delete(handle)

# Shutdown
orchestrator.shutdown()
```

### 2. Vector Search with Jina v3

```python
from query_aware_embeddings import JinaV3Embedder

# Initialize embedder
embedder = JinaV3Embedder(
    model_name="jinaai/jina-embeddings-v3",
    output_dim=1024  # Matryoshka 1024D
)

# Store document with embedding
text = "Analysis of APT29 malware campaign targeting government networks..."
embedding = embedder.encode_document([text])[0]

handle = orchestrator.store(
    data={'text': text, 'embedding': embedding},
    content_type=ContentType.EMBEDDING,
    metadata={'source': 'threat_intel', 'date': '2024-01-15'}
)

# Search by semantic similarity
query = "advanced persistent threat government attack"
query_embedding = embedder.encode_query([query])[0]

results = orchestrator.vector_search(
    query_embedding=query_embedding,
    content_type=ContentType.EMBEDDING,
    top_k=10
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.content['text'][:200]}...")
    print()
```

### 3. Caching Strategy

```python
# Store with automatic caching
handle = orchestrator.store(
    data={'expensive': 'computation_result'},
    content_type=ContentType.CACHE,
    key='analysis:report_123',
    ttl=3600
)

# First retrieve - cache miss, reads from primary
result = orchestrator.retrieve(handle, use_cache=True)

# Second retrieve - cache hit, served from Redis (< 1ms)
result = orchestrator.retrieve(handle, use_cache=True)

# Check cache stats
stats = orchestrator.get_stats()
hit_rate = stats['orchestrator']['cache_hits'] / (
    stats['orchestrator']['cache_hits'] +
    stats['orchestrator']['cache_misses']
)
print(f"Cache hit rate: {hit_rate * 100:.1f}%")
```

### 4. Health Monitoring

```python
# Check health of all backends
health_status = orchestrator.health_check_all()

for backend, (is_healthy, message) in health_status.items():
    status = "✓" if is_healthy else "✗"
    print(f"{status} {backend}: {message}")

# Output:
# ✓ postgresql: PostgreSQL connection healthy
# ✓ redis: Redis connection healthy
# ✓ vector: Qdrant healthy (1 collections)
# ✓ sqlite: SQLite connection healthy

# Get detailed statistics
stats = orchestrator.get_stats()

print("\nPostgreSQL:")
print(f"  Items: {stats['backends']['postgresql']['total_items']:,}")
print(f"  Size: {stats['backends']['postgresql']['total_size_bytes'] / 1024 / 1024:.1f} MB")
print(f"  Avg access: {stats['backends']['postgresql']['avg_access_time_ms']:.2f} ms")

print("\nRedis:")
print(f"  Items: {stats['backends']['redis']['total_items']:,}")
print(f"  Hit rate: {stats['backends']['redis']['custom_metrics']['hit_rate'] * 100:.1f}%")

print("\nQdrant:")
print(f"  Vectors: {stats['backends']['vector']['custom_metrics']['vectors_count']:,}")
print(f"  Avg search: {stats['backends']['vector']['avg_access_time_ms']:.2f} ms")
```

### 5. Backup and Recovery

```python
import os
from datetime import datetime

# Backup all backends
backup_dir = f"/backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

backup_status = orchestrator.backup_all(backup_dir)

for backend, success in backup_status.items():
    status = "✓" if success else "✗"
    print(f"{status} {backend} backup: {'Success' if success else 'Failed'}")

# Output:
# ✓ postgresql backup: Success
# ✓ redis backup: Success
# ✓ vector backup: Success
# ✓ sqlite backup: Success

print(f"\nBackups saved to: {backup_dir}")
```

### 6. Optimization

```python
# Run optimization on all backends
optimization_status = orchestrator.optimize_all()

for backend, success in optimization_status.items():
    status = "✓" if success else "✗"
    print(f"{status} {backend} optimization: {'Complete' if success else 'Failed'}")

# Output:
# ✓ postgresql optimization: Complete (VACUUM ANALYZE)
# ✓ redis optimization: Complete (Expired 127 keys)
# ✓ vector optimization: Complete (Reindex triggered)
# ✓ sqlite optimization: Complete (VACUUM, FTS rebuild)
```

---

## Configuration

### Environment Variables

```bash
# PostgreSQL
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=ai_engine
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=password

# Redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

# Qdrant
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_COLLECTION=embeddings

# SQLite
export SQLITE_PATH=/dev/shm/lat5150_ramdisk.db
```

### Configuration File (YAML)

```yaml
# storage_config.yaml
postgresql:
  host: localhost
  port: 5432
  database: ai_engine
  user: postgres
  password: ${POSTGRES_PASSWORD}
  min_connections: 2
  max_connections: 10

redis:
  host: localhost
  port: 6379
  db: 0
  max_connections: 50
  key_prefix: "lat5150:"

qdrant:
  host: localhost
  port: 6333
  collection_name: jina_v3_embeddings
  vector_size: 1024
  distance: cosine
  use_quantization: true
  hnsw_m: 32
  hnsw_ef_construct: 200

sqlite:
  db_path: /dev/shm/lat5150_ramdisk.db
  use_wal: true
  cache_size: 2000
  enable_fts: true

orchestrator:
  enable_caching: true
  enable_health_checks: true
  health_check_interval: 60
  enable_auto_optimization: true
  optimization_interval: 3600
```

### Loading Configuration

```python
import yaml
from storage_orchestrator import OrchestratorConfig, StorageOrchestrator

# Load from YAML
with open('storage_config.yaml') as f:
    config_dict = yaml.safe_load(f)

config = OrchestratorConfig(
    postgresql_config=config_dict['postgresql'],
    redis_config=config_dict['redis'],
    qdrant_config=config_dict['qdrant'],
    sqlite_config=config_dict['sqlite'],
    **config_dict['orchestrator']
)

orchestrator = StorageOrchestrator(config)
orchestrator.initialize()
```

---

## Usage Examples

### Example 1: Storing and Retrieving Conversations

```python
# Store conversation
conversation_data = {
    'title': 'Cyber Threat Investigation',
    'user_id': 'analyst_007',
    'created_at': '2024-01-15T10:00:00',
    'status': 'active'
}

handle = orchestrator.store(
    data=conversation_data,
    content_type=ContentType.CONVERSATION,
    metadata={'priority': 'high', 'classification': 'confidential'}
)

# Retrieve conversation (served from cache after first access)
conversation = orchestrator.retrieve(handle)

print(f"Conversation: {conversation['title']}")
print(f"User: {conversation['user_id']}")
print(f"Status: {conversation['status']}")
```

### Example 2: Semantic Search with RAG

```python
from rag_orchestrator import RAGOrchestrator

# Initialize RAG with Jina v3
rag = RAGOrchestrator(preset="jina_high_accuracy")
storage = create_default_orchestrator()

# Index document
doc_text = """
Advanced Persistent Threat (APT29) has been observed using
sophisticated malware to target government networks. The attack
vector includes spear-phishing emails with malicious attachments.
"""

doc_id = rag.index_document(doc_text, doc_id="threat_report_001")

# Search
query = "How does APT29 attack government systems?"
results = rag.search(query, top_k=5)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result.score:.3f}")
    print(f"   {result.text[:200]}...")
```

### Example 3: Distributed Locking

```python
# Acquire lock for critical section
lock_name = "process:malware_analysis"

lock = orchestrator.backends[StorageType.REDIS].acquire_lock(
    lock_name=lock_name,
    timeout=30,  # Lock expires after 30s
    blocking=True
)

if lock:
    try:
        # Critical section - only one process at a time
        print("Processing malware sample...")
        # ... perform analysis ...

    finally:
        # Always release lock
        orchestrator.backends[StorageType.REDIS].release_lock(lock)
else:
    print("Could not acquire lock")
```

### Example 4: Audit Logging

```python
# Log user action
audit_entry = {
    'action': 'document_accessed',
    'entity_type': 'threat_report',
    'entity_id': 'report_12345',
    'user_id': 'analyst_007',
    'details': {
        'ip_address': '10.0.1.50',
        'user_agent': 'Mozilla/5.0...',
        'classification': 'confidential'
    }
}

handle = orchestrator.store(
    data=audit_entry,
    content_type=ContentType.AUDIT,
    metadata={'timestamp': datetime.now().isoformat()}
)

# Search audit logs
results = orchestrator.search(
    query="document_accessed",
    content_type=ContentType.AUDIT,
    storage_types=[StorageType.SQLITE],
    limit=100
)

print(f"Found {len(results)} audit entries")
for result in results:
    print(f"  {result.content['action']} by {result.content['user_id']}")
```

---

## Best Practices

### 1. Choose the Right Backend

| Data Type | Primary Backend | Cache | Reasoning |
|-----------|----------------|-------|-----------|
| Conversations | PostgreSQL | Redis | Need ACID, complex queries, caching for hot data |
| Messages | PostgreSQL | Redis | Same as conversations, high read frequency |
| Documents | PostgreSQL | - | Large text, full-text search, low read frequency |
| Embeddings | Qdrant | Redis | Vector similarity, hot queries cached |
| Sessions | Redis | - | Temporary, fast access, TTL expiration |
| Checkpoints | SQLite | - | Large files, periodic access, local storage |
| Audit Logs | SQLite | - | Write-heavy, sequential reads |

### 2. Caching Strategy

```python
# Cache hot data with appropriate TTL
policies = {
    # Frequently accessed, cache for 1 hour
    ContentType.CONVERSATION: StoragePolicy(
        primary_backend=StorageType.POSTGRESQL,
        cache_backend=StorageType.REDIS,
        cache_ttl=3600
    ),

    # Very hot data, cache for 30 minutes
    ContentType.SESSION: StoragePolicy(
        primary_backend=StorageType.REDIS,
        cache_ttl=1800
    ),

    # Rarely accessed, no caching
    ContentType.AUDIT: StoragePolicy(
        primary_backend=StorageType.SQLITE,
        cache_backend=None
    )
}
```

### 3. Connection Pooling

```python
# Optimize PostgreSQL pool size based on workload
postgresql_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_engine',
    'user': 'postgres',
    'password': 'password',

    # For read-heavy workload
    'min_connections': 5,
    'max_connections': 20,

    # For write-heavy workload
    # 'min_connections': 2,
    # 'max_connections': 10,
}
```

### 4. Vector Search Optimization

```python
# Optimize Qdrant for your use case
qdrant_config = {
    'host': 'localhost',
    'port': 6333,
    'collection_name': 'embeddings',

    # Jina v3 with Matryoshka 1024D
    'vector_size': 1024,

    # Cosine similarity for normalized vectors
    'distance': 'cosine',

    # Enable quantization for 4x memory reduction
    'use_quantization': True,

    # HNSW tuning
    'hnsw_m': 32,              # Higher = better recall, more memory
    'hnsw_ef_construct': 200,  # Higher = better quality index
    'hnsw_ef_search': 128      # Higher = better recall, slower search
}

# For maximum accuracy (95-97%)
qdrant_config.update({
    'vector_size': 2084,  # Full Jina v3
    'hnsw_m': 64,
    'hnsw_ef_construct': 400,
    'hnsw_ef_search': 256
})

# For maximum speed
qdrant_config.update({
    'vector_size': 512,  # Reduced Matryoshka
    'hnsw_m': 16,
    'hnsw_ef_construct': 100,
    'hnsw_ef_search': 64
})
```

### 5. Health Monitoring

```python
import logging

# Enable health monitoring
config = OrchestratorConfig(
    enable_health_checks=True,
    health_check_interval=60  # Check every minute
)

# Log health issues
logging.basicConfig(level=logging.WARNING)

# The orchestrator will automatically log unhealthy backends:
# WARNING: Unhealthy backends: ['postgresql']
# WARNING: postgresql health check failed: connection refused
```

### 6. Backup Strategy

```python
import schedule
import time

def backup_job():
    """Daily backup at 2 AM"""
    backup_dir = f"/backups/{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(backup_dir, exist_ok=True)

    logger.info(f"Starting backup to {backup_dir}")
    results = orchestrator.backup_all(backup_dir)

    if all(results.values()):
        logger.info("Backup completed successfully")
    else:
        logger.error(f"Backup failed: {results}")

# Schedule daily backup
schedule.every().day.at("02:00").do(backup_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Performance Optimization

### PostgreSQL Optimization

```sql
-- Enable shared buffers (25% of RAM)
ALTER SYSTEM SET shared_buffers = '4GB';

-- Increase work memory for complex queries
ALTER SYSTEM SET work_mem = '64MB';

-- Enable parallel query execution
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Optimize for SSD
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();
```

### Redis Optimization

```bash
# redis.conf

# Use all available memory (minus 20% for overhead)
maxmemory 8gb

# LRU eviction policy
maxmemory-policy allkeys-lru

# Enable persistence (RDB + AOF)
save 900 1
save 300 10
save 60 10000

appendonly yes
appendfsync everysec

# Disable slow operations
rename-command FLUSHDB ""
rename-command FLUSHALL ""
```

### Qdrant Optimization

```python
# Optimize collection after bulk insert
qdrant_backend.optimize()

# Use batch operations
handles = qdrant_backend.batch_store_embeddings(
    texts=texts,          # 1000 documents
    embeddings=embeddings,
    content_type=ContentType.EMBEDDING
)

# Enable quantization for large collections
config = {
    'use_quantization': True,  # 4x memory reduction
    'quantization_config': {
        'scalar': 'int8',
        'quantile': 0.99,
        'always_ram': True
    }
}
```

### SQLite Optimization

```python
# Use RAM disk for checkpoints
config = {
    'db_path': '/dev/shm/checkpoints.db',
    'use_wal': True,
    'cache_size': 10000,  # 10MB
    'mmap_size': 268435456  # 256MB mmap
}

# Or tmpfs mount
# sudo mount -t tmpfs -o size=1G tmpfs /mnt/tmpfs
config = {
    'db_path': '/mnt/tmpfs/checkpoints.db'
}
```

---

## Troubleshooting

### Issue: PostgreSQL Connection Refused

**Symptoms:**
```
ERROR: Failed to connect to PostgreSQL: connection refused
```

**Solutions:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Check connection
psql -h localhost -U postgres -d ai_engine

# Verify pg_hba.conf allows connections
sudo vim /etc/postgresql/*/main/pg_hba.conf
# Add: host all all 127.0.0.1/32 md5
```

### Issue: Redis Memory Limit Exceeded

**Symptoms:**
```
WARNING: Redis memory limit exceeded, evicting keys
```

**Solutions:**
```python
# Check memory usage
redis_backend = orchestrator.backends[StorageType.REDIS]
stats = redis_backend.get_stats()
print(f"Memory: {stats.total_size_bytes / 1024 / 1024:.1f} MB")

# Clear cache
redis_backend.cache_clear()

# Increase Redis memory limit
# redis.conf: maxmemory 16gb

# Or use LRU eviction
# redis.conf: maxmemory-policy allkeys-lru
```

### Issue: Qdrant Search Slow

**Symptoms:**
```
Average search time > 100ms
```

**Solutions:**
```python
# Check collection stats
qdrant_backend = orchestrator.backends[StorageType.VECTOR]
stats = qdrant_backend.get_stats()
print(f"Vectors: {stats.custom_metrics['vectors_count']:,}")
print(f"Indexed: {stats.custom_metrics['indexed_vectors_count']:,}")

# Optimize collection
qdrant_backend.optimize()

# Reduce ef_search for faster (less accurate) results
config['hnsw_ef_search'] = 64  # Default: 128

# Enable quantization
config['use_quantization'] = True
```

### Issue: SQLite Database Locked

**Symptoms:**
```
ERROR: database is locked
```

**Solutions:**
```python
# Enable WAL mode (default in our config)
config = {
    'db_path': '/path/to/db.sqlite',
    'use_wal': True,  # Allows concurrent reads
    'journal_mode': 'WAL'
}

# Increase busy timeout
sqlite_backend.conn.execute("PRAGMA busy_timeout=10000")  # 10 seconds
```

### Issue: Cache Hit Rate Too Low

**Symptoms:**
```
Cache hit rate: 15%
```

**Solutions:**
```python
# Increase TTL for hot data
policies[ContentType.CONVERSATION].cache_ttl = 7200  # 2 hours

# Pre-warm cache
hot_conversations = orchestrator.search(
    query="*",
    content_type=ContentType.CONVERSATION,
    limit=1000
)

for result in hot_conversations:
    # This will cache each result
    orchestrator.retrieve(result.handle, use_cache=True)

# Check hit rate
stats = orchestrator.get_stats()
hit_rate = stats['orchestrator']['cache_hits'] / (
    stats['orchestrator']['cache_hits'] +
    stats['orchestrator']['cache_misses']
)
print(f"New hit rate: {hit_rate * 100:.1f}%")
```

---

## Summary

The **Unified Storage System** provides a production-ready, intelligent storage layer for the LAT5150DRVMIL AI engine with:

✅ **4 Storage Backends** - PostgreSQL, Redis, Qdrant, SQLite
✅ **Automatic Routing** - Content-type based backend selection
✅ **Intelligent Caching** - Redis-based caching with TTL
✅ **Vector Search** - Jina v3 embeddings with Qdrant HNSW
✅ **Health Monitoring** - Continuous health checks
✅ **Backup/Recovery** - Coordinated backup across backends
✅ **ACID Transactions** - Where applicable (PostgreSQL, SQLite)
✅ **95-97% Accuracy** - With Jina v3 high-accuracy preset
✅ **Sub-millisecond** - Redis cache access times
✅ **4x Compression** - Qdrant INT8 quantization

**Production deployments should:**
- Enable health monitoring and alerting
- Configure daily backups
- Optimize backend configurations for workload
- Monitor cache hit rates
- Use connection pooling appropriately
- Enable WAL mode for SQLite
- Use quantization for large vector collections

For questions or issues, consult the troubleshooting section or check backend-specific logs.

---

**Version:** 1.0.0
**Last Updated:** 2025-11-13
**Status:** Production Ready ✓
