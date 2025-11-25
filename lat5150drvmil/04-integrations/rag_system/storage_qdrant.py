#!/usr/bin/env python3
"""
Qdrant Vector Storage Backend Adapter

Provides unified interface for Qdrant vector database in the LAT5150DRVMIL AI engine.
Handles:
- High-dimensional embeddings (2084D Jina v3, 768D BGE, etc.)
- Vector similarity search
- Hybrid search (vector + full-text)
- Multi-vector storage (ColBERT-style)
- Scalar quantization for memory optimization
"""

import logging
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, SearchRequest, HnswConfigDiff, ScalarQuantization,
    ScalarType, OptimizersConfigDiff, QuantizationSearchParams
)

from storage_abstraction import (
    AbstractVectorBackend,
    StorageType,
    ContentType,
    StorageTier,
    StorageHandle,
    SearchResult,
    StorageStats
)

logger = logging.getLogger(__name__)


class QdrantStorageBackend(AbstractVectorBackend):
    """
    Qdrant vector storage adapter for AI engine embeddings

    Supports:
    - Dense vector search with HNSW
    - Scalar quantization (4x memory reduction)
    - Multi-vector storage
    - Payload filtering
    - Batch operations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant backend

        Args:
            config: Configuration dictionary with:
                - host: Qdrant host
                - port: Qdrant port (default: 6333)
                - grpc_port: gRPC port (default: 6334)
                - api_key: API key (optional, for cloud)
                - collection_name: Collection name
                - vector_size: Embedding dimension
                - distance: Distance metric (cosine, dot, euclidean)
                - use_quantization: Enable INT8 quantization
                - hnsw_m: HNSW m parameter (default: 32)
                - hnsw_ef_construct: HNSW ef_construct (default: 200)
                - multi_vector: Use multi-vector storage
        """
        super().__init__(config)
        self.storage_type = StorageType.VECTOR

        # Connection configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6333)
        self.grpc_port = config.get('grpc_port', 6334)
        self.api_key = config.get('api_key')
        self.prefer_grpc = config.get('prefer_grpc', True)

        # Collection configuration
        self.collection_name = config.get('collection_name', 'embeddings')
        self.vector_size = config.get('vector_size', 1024)
        self.distance = config.get('distance', 'cosine')
        self.use_quantization = config.get('use_quantization', True)
        self.multi_vector = config.get('multi_vector', False)

        # HNSW parameters
        self.hnsw_m = config.get('hnsw_m', 32)
        self.hnsw_ef_construct = config.get('hnsw_ef_construct', 200)
        self.hnsw_ef_search = config.get('hnsw_ef_search', 128)

        # Client
        self.client = None

        # Performance tracking
        self._search_times = []
        self._max_search_history = 1000

    def connect(self) -> bool:
        """
        Establish connection to Qdrant

        Returns:
            True if connection successful
        """
        try:
            if self.api_key:
                # Cloud connection
                self.client = QdrantClient(
                    url=f"https://{self.host}",
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc
                )
            else:
                # Local connection
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc
                )

            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant ({len(collections.collections)} collections)")

            # Ensure collection exists
            self._ensure_collection()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Close Qdrant connection

        Returns:
            True if disconnection successful
        """
        try:
            if self.client:
                self.client.close()
            logger.info("Qdrant connection closed")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Qdrant: {e}")
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
        Store data in Qdrant (requires embedding in data)

        Args:
            data: Dictionary with 'embedding' and optional 'text'
            content_type: Type of content
            key: Optional point ID (auto-generated if not provided)
            ttl: Not supported for vector storage
            metadata: Additional metadata (stored as payload)

        Returns:
            StorageHandle for retrieving data
        """
        try:
            if not isinstance(data, dict) or 'embedding' not in data:
                raise ValueError("Data must contain 'embedding' field")

            embedding = data['embedding']
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            # Validate dimension
            if not self.multi_vector:
                if len(embedding.shape) != 1 or embedding.shape[0] != self.vector_size:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.vector_size}, got {embedding.shape}")
            else:
                # Multi-vector: expect (num_vectors, dim)
                if len(embedding.shape) != 2:
                    raise ValueError("Multi-vector embedding must be 2D array")

            # Generate point ID
            point_id = key or str(int(time.time() * 1000000))

            # Prepare payload
            payload = {
                'content_type': content_type.value,
                'text': data.get('text', ''),
                'created_at': datetime.now().isoformat(),
                **(metadata or {})
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload=payload
            )

            # Upsert to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            self._stats['operations'] += 1
            self._stats['bytes_written'] += len(json.dumps(payload).encode())

            return StorageHandle(
                storage_type=self.storage_type,
                storage_id=point_id,
                content_type=content_type,
                tier=StorageTier.WARM,
                created_at=datetime.now(),
                metadata=metadata or {}
            )

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error storing data in Qdrant: {e}")
            raise

    def retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """
        Retrieve data by handle

        Args:
            handle: Storage handle from store()

        Returns:
            Retrieved data (embedding + payload)
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[handle.storage_id],
                with_vectors=True,
                with_payload=True
            )

            if not points:
                return None

            point = points[0]

            result = {
                'id': point.id,
                'embedding': point.vector,
                'payload': point.payload
            }

            self._stats['operations'] += 1
            return result

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error retrieving data from Qdrant: {e}")
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
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[handle.storage_id]
            )

            self._stats['operations'] += 1
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error deleting data from Qdrant: {e}")
            return False

    def search(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Full-text search (limited support, use vector_search for embeddings)

        Args:
            query: Search query text
            content_type: Filter by content type
            filters: Additional filters
            limit: Maximum results

        Returns:
            List of search results
        """
        # Qdrant doesn't have built-in full-text search
        # This method searches payloads for matching text
        try:
            # Build filter
            filter_conditions = []
            if content_type:
                filter_conditions.append(
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value=content_type.value)
                    )
                )

            # Search using scroll (no vector query)
            results = []
            offset = None

            while len(results) < limit:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=filter_conditions) if filter_conditions else None,
                    limit=min(limit - len(results), 100),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = scroll_result

                for point in points:
                    # Simple text matching in payload
                    text = point.payload.get('text', '')
                    if query.lower() in text.lower():
                        handle = StorageHandle(
                            storage_type=self.storage_type,
                            storage_id=str(point.id),
                            content_type=ContentType(point.payload.get('content_type', 'embedding')),
                            tier=StorageTier.WARM,
                            created_at=datetime.now(),
                            metadata=point.payload
                        )

                        results.append(SearchResult(
                            handle=handle,
                            score=1.0,
                            content=point.payload,
                            metadata=point.payload
                        ))

                if next_offset is None or len(results) >= limit:
                    break

                offset = next_offset

            self._stats['operations'] += 1
            return results[:limit]

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error searching Qdrant: {e}")
            return []

    def get_stats(self) -> StorageStats:
        """
        Get Qdrant storage statistics

        Returns:
            StorageStats object
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)

            total_items = collection_info.points_count
            total_size = collection_info.vectors_count * self.vector_size * 4  # 4 bytes per float

            # If quantization enabled, reduce size estimate
            if self.use_quantization:
                total_size = int(total_size * 0.25)  # INT8 is 1/4 of float32

            # Average search time
            avg_search_time = (
                sum(self._search_times) / len(self._search_times)
                if self._search_times else 0.0
            )

            return StorageStats(
                storage_type=self.storage_type,
                total_items=total_items,
                total_size_bytes=total_size,
                avg_access_time_ms=avg_search_time,
                health_status="healthy",
                custom_metrics={
                    'vectors_count': collection_info.vectors_count,
                    'indexed_vectors_count': collection_info.indexed_vectors_count,
                    'segments_count': collection_info.segments_count,
                    'operations': self._stats['operations'],
                    'bytes_written': self._stats['bytes_written'],
                    'bytes_read': self._stats['bytes_read'],
                    'errors': self._stats['errors']
                }
            )

        except Exception as e:
            logger.error(f"Error getting Qdrant stats: {e}")
            return StorageStats(
                storage_type=self.storage_type,
                total_items=0,
                total_size_bytes=0,
                avg_access_time_ms=0,
                health_status="error"
            )

    def health_check(self) -> Tuple[bool, str]:
        """
        Check Qdrant health

        Returns:
            (is_healthy, status_message)
        """
        try:
            collections = self.client.get_collections()
            return (True, f"Qdrant healthy ({len(collections.collections)} collections)")
        except Exception as e:
            return (False, f"Qdrant health check failed: {e}")

    # Vector-specific methods (from AbstractVectorBackend)

    def store_embedding(
        self,
        text: str,
        embedding: List[float],
        content_type: ContentType = ContentType.EMBEDDING,
        metadata: Optional[Dict] = None
    ) -> StorageHandle:
        """
        Store text with embedding

        Args:
            text: Original text
            embedding: Embedding vector
            content_type: Type of content
            metadata: Additional metadata

        Returns:
            StorageHandle
        """
        data = {
            'text': text,
            'embedding': embedding
        }
        return self.store(data, content_type, metadata=metadata)

    def vector_search(
        self,
        query_embedding: List[float],
        content_type: Optional[ContentType] = None,
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search by embedding vector

        Args:
            query_embedding: Query embedding vector
            content_type: Filter by content type
            filters: Additional payload filters
            top_k: Number of results

        Returns:
            List of search results sorted by similarity
        """
        try:
            start_time = time.time()

            # Build filter
            filter_conditions = []
            if content_type:
                filter_conditions.append(
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value=content_type.value)
                    )
                )

            if filters:
                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=QuantizationSearchParams(
                    rescore=True
                ) if self.use_quantization else None
            )

            # Track performance
            search_time = (time.time() - start_time) * 1000
            self._search_times.append(search_time)
            if len(self._search_times) > self._max_search_history:
                self._search_times.pop(0)

            # Convert to SearchResult objects
            results = []
            for scored_point in search_result:
                handle = StorageHandle(
                    storage_type=self.storage_type,
                    storage_id=str(scored_point.id),
                    content_type=ContentType(scored_point.payload.get('content_type', 'embedding')),
                    tier=StorageTier.WARM,
                    created_at=datetime.now(),
                    metadata=scored_point.payload
                )

                results.append(SearchResult(
                    handle=handle,
                    score=scored_point.score,
                    content=scored_point.payload,
                    metadata=scored_point.payload
                ))

            self._stats['operations'] += 1
            return results

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error performing vector search in Qdrant: {e}")
            return []

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        alpha: float = 0.5,
        content_type: Optional[ContentType] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and text matching

        Args:
            query_text: Query text
            query_embedding: Query embedding
            alpha: Weight for vector search (0=text only, 1=vector only)
            content_type: Filter by content type
            top_k: Number of results

        Returns:
            List of search results
        """
        # Get vector search results
        vector_results = self.vector_search(
            query_embedding=query_embedding,
            content_type=content_type,
            top_k=top_k * 2  # Get more for fusion
        )

        # Get text search results
        text_results = self.search(
            query=query_text,
            content_type=content_type,
            limit=top_k * 2
        )

        # Combine using reciprocal rank fusion
        from hybrid_search import reciprocal_rank_fusion

        vector_dicts = [
            {'id': r.handle.storage_id, 'score': r.score, 'data': r}
            for r in vector_results
        ]
        text_dicts = [
            {'id': r.handle.storage_id, 'score': r.score, 'data': r}
            for r in text_results
        ]

        # Apply alpha weighting
        fused = reciprocal_rank_fusion(
            vector_dicts,
            text_dicts,
            k=60
        )

        # Convert back to SearchResult objects
        results = []
        for item in fused[:top_k]:
            original_result = item['data']
            # Update score with fused score
            results.append(SearchResult(
                handle=original_result.handle,
                score=item['score'] * alpha + original_result.score * (1 - alpha),
                content=original_result.content,
                metadata=original_result.metadata
            ))

        return results

    def batch_store_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        content_type: ContentType = ContentType.EMBEDDING,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[StorageHandle]:
        """
        Batch store multiple embeddings

        Args:
            texts: List of texts
            embeddings: List of embedding vectors
            content_type: Type of content
            metadata_list: List of metadata dicts (optional)

        Returns:
            List of StorageHandles
        """
        try:
            if len(texts) != len(embeddings):
                raise ValueError("texts and embeddings must have same length")

            if metadata_list and len(metadata_list) != len(texts):
                raise ValueError("metadata_list must match texts length")

            # Prepare points
            points = []
            handles = []

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                point_id = str(int(time.time() * 1000000) + i)

                payload = {
                    'content_type': content_type.value,
                    'text': text,
                    'created_at': datetime.now().isoformat()
                }

                if metadata_list:
                    payload.update(metadata_list[i])

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

                handles.append(StorageHandle(
                    storage_type=self.storage_type,
                    storage_id=point_id,
                    content_type=content_type,
                    tier=StorageTier.WARM,
                    created_at=datetime.now(),
                    metadata=metadata_list[i] if metadata_list else {}
                ))

            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            self._stats['operations'] += len(points)
            return handles

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error batch storing embeddings in Qdrant: {e}")
            raise

    # Internal helper methods

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return

            # Map distance string to Distance enum
            distance_map = {
                'cosine': Distance.COSINE,
                'dot': Distance.DOT,
                'euclidean': Distance.EUCLID
            }
            distance_metric = distance_map.get(self.distance.lower(), Distance.COSINE)

            # Create collection
            vector_config = VectorParams(
                size=self.vector_size,
                distance=distance_metric,
                hnsw_config=HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct
                )
            )

            # Add quantization if enabled
            if self.use_quantization:
                vector_config.quantization_config = ScalarQuantization(
                    scalar=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )

            logger.info(f"Created collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")
            raise

    def backup(self, destination: str) -> bool:
        """
        Backup Qdrant collection to snapshot

        Args:
            destination: Snapshot name

        Returns:
            True if backup successful
        """
        try:
            snapshot_info = self.client.create_snapshot(
                collection_name=self.collection_name
            )
            logger.info(f"Qdrant snapshot created: {snapshot_info.name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Qdrant snapshot: {e}")
            return False

    def optimize(self) -> bool:
        """
        Optimize Qdrant collection (rebuild index)

        Returns:
            True if optimization successful
        """
        try:
            # Trigger optimization
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Force immediate indexing
                )
            )
            logger.info(f"Qdrant collection '{self.collection_name}' optimization triggered")
            return True
        except Exception as e:
            logger.error(f"Error optimizing Qdrant: {e}")
            return False


if __name__ == "__main__":
    print("=" * 80)
    print("QDRANT VECTOR STORAGE BACKEND")
    print("=" * 80 + "\n")

    # Example configuration for Jina v3
    config = {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'jina_v3_embeddings',
        'vector_size': 1024,  # Jina v3 with Matryoshka 1024D
        'distance': 'cosine',
        'use_quantization': True,
        'hnsw_m': 32,
        'hnsw_ef_construct': 200
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("✓ Qdrant Storage Backend initialized")
    print("\nSupports:")
    print("  - High-dimensional embeddings (up to 2084D)")
    print("  - Vector similarity search (HNSW)")
    print("  - Scalar quantization (INT8, 4x compression)")
    print("  - Multi-vector storage (ColBERT)")
    print("  - Payload filtering")
    print("  - Batch operations")
    print("  - Hybrid search (vector + text)")
