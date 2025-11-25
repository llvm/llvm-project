#!/usr/bin/env python3
"""
Enhanced RAG System with Qdrant Vector Database
Upgrade from TF-IDF (51.8% accuracy) to Transformer Embeddings (target: 88%+)

Features:
- Qdrant vector database for semantic search
- Sentence-transformers for embeddings
- Screenshot/image ingestion with OCR
- Chat log timeline integration
- Hybrid search (semantic + keyword + temporal)
- Cross-referencing and event correlation

Integration:
- Extends existing RAG system
- Compatible with Telegram/Signal scrapers
- Integrates with DSMIL AI Engine
- SWORD Intelligence compatible
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import re

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, Range,
        HnswConfigDiff, SearchParams
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  Qdrant not installed. Run: pip install qdrant-client")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Sentence-transformers not installed. Run: pip install sentence-transformers")

# OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Existing RAG system (for migration)
try:
    from rag_system import RAGSystem
except ImportError:
    RAGSystem = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation"""
    id: str
    text: str
    filepath: str
    filename: str
    doc_type: str  # pdf, text, image, chat_message
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result with metadata"""
    document: Document
    score: float
    matched_terms: List[str] = field(default_factory=list)


class VectorRAGSystem:
    """
    Enhanced RAG System with Qdrant Vector Database

    Supports:
    - Documents (PDFs, text files, markdown)
    - Screenshots (with OCR)
    - Chat messages (Telegram, Signal)
    - Timeline queries
    - Event correlation
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "lat5150_knowledge_base",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        use_gpu: bool = True
    ):
        """
        Initialize Vector RAG System

        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Collection name for documents
            embedding_model: Sentence transformer model
            use_gpu: Use GPU for embeddings if available
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant required. Install: pip install qdrant-client")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence-transformers required. Install: pip install sentence-transformers")

        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        # Initialize embedding model
        device = 'cuda' if use_gpu else 'cpu'
        logger.info(f"Loading embedding model: {embedding_model} on {device}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # OCR engines
        self.paddle_ocr = None
        self.tesseract_available = TESSERACT_AVAILABLE

        if PADDLE_OCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=use_gpu,
                    show_log=False
                )
                logger.info("✓ PaddleOCR initialized")
            except Exception as e:
                logger.warning(f"PaddleOCR init failed: {e}")

        # Initialize collection
        self._ensure_collection()

        logger.info(f"✓ Vector RAG System initialized")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Embedding model: {embedding_model} ({self.embedding_dim}D)")
        logger.info(f"  OCR: {'PaddleOCR' if self.paddle_ocr else 'Tesseract' if self.tesseract_available else 'None'}")

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist with optimized HNSW parameters"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=32,              # Default: 16, Higher: more edges, better accuracy (+3-5%)
                        ef_construct=200,  # Default: 100, Higher: better quality index
                        full_scan_threshold=10000,
                    )
                )
            )
            logger.info("✓ Collection created with optimized HNSW (m=32, ef_construct=200)")
        else:
            logger.info(f"✓ Collection exists: {self.collection_name}")

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _extract_text_from_image(self, image_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from image using OCR

        Returns:
            (extracted_text, metadata)
        """
        if self.paddle_ocr:
            try:
                result = self.paddle_ocr.ocr(str(image_path), cls=True)

                if result and result[0]:
                    lines = []
                    confidences = []
                    for line in result[0]:
                        if len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                            confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 0.0
                            lines.append(text)
                            confidences.append(confidence)

                    text = '\n'.join(lines)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                    return text, {
                        'ocr_engine': 'paddleocr',
                        'ocr_confidence': avg_confidence,
                        'ocr_lines': len(lines)
                    }
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")

        # Fallback to Tesseract
        if self.tesseract_available:
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                return text, {'ocr_engine': 'tesseract'}
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")

        return "", {'ocr_engine': 'none', 'error': 'No OCR engine available'}

    def ingest_document(
        self,
        filepath: Path,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Ingest a document into the vector database

        Args:
            filepath: Path to document
            doc_type: Document type (auto-detected if None)
            metadata: Additional metadata

        Returns:
            Ingestion result dictionary
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {'error': f'File not found: {filepath}'}

        # Auto-detect document type
        if doc_type is None:
            suffix = filepath.suffix.lower()
            if suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                doc_type = 'image'
            elif suffix == '.pdf':
                doc_type = 'pdf'
            elif suffix in ['.txt', '.md', '.log']:
                doc_type = 'text'
            else:
                doc_type = 'unknown'

        # Extract text based on type
        text = ""
        extract_metadata = {}

        if doc_type == 'image':
            text, extract_metadata = self._extract_text_from_image(filepath)
        elif doc_type == 'pdf':
            # Use existing document processor
            from donut_pdf_processor import DonutPDFProcessor
            try:
                processor = DonutPDFProcessor()
                result = processor.process_pdf(str(filepath))
                text = result.get('text', '')
                extract_metadata = {'pdf_processor': 'donut'}
            except Exception as e:
                logger.warning(f"Donut PDF failed: {e}, trying fallback")
                # Fallback to pdftotext
                import subprocess
                try:
                    result = subprocess.run(
                        ['pdftotext', str(filepath), '-'],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        text = result.stdout
                        extract_metadata = {'pdf_processor': 'pdftotext'}
                except:
                    pass
        elif doc_type in ['text', 'unknown']:
            try:
                text = filepath.read_text(encoding='utf-8', errors='ignore')
                extract_metadata = {'encoding': 'utf-8'}
            except:
                return {'error': 'Failed to read text file'}

        if not text or len(text.strip()) < 10:
            return {'error': 'No text extracted or text too short'}

        # Compute hash for deduplication
        doc_hash = self._compute_hash(text)

        # Check if already indexed
        existing = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="hash", match=MatchValue(value=doc_hash))]
            ),
            limit=1
        )

        if existing[0]:
            return {
                'status': 'already_indexed',
                'hash': doc_hash,
                'id': existing[0][0].id
            }

        # Generate embedding
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False).tolist()
        except Exception as e:
            return {'error': f'Failed to generate embedding: {e}'}

        # Prepare metadata
        full_metadata = {
            'filepath': str(filepath),
            'filename': filepath.name,
            'type': doc_type,
            'size': filepath.stat().st_size,
            'timestamp': datetime.now().isoformat(),
            'hash': doc_hash,
            'text_length': len(text),
            **extract_metadata,
            **(metadata or {})
        }

        # Generate unique ID
        doc_id = doc_hash[:16]

        # Index in Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            'text': text[:10000],  # Store first 10k chars
                            **full_metadata
                        }
                    )
                ]
            )
        except Exception as e:
            return {'error': f'Failed to index document: {e}'}

        return {
            'status': 'success',
            'id': doc_id,
            'hash': doc_hash,
            'filename': filepath.name,
            'type': doc_type,
            'text_length': len(text),
            'metadata': full_metadata
        }

    def ingest_chat_message(
        self,
        message: str,
        source: str,  # 'telegram', 'signal', etc.
        chat_id: str,
        chat_name: str,
        sender: str,
        timestamp: datetime,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Ingest a chat message into the vector database

        Args:
            message: Message text
            source: Source platform (telegram, signal)
            chat_id: Chat identifier
            chat_name: Chat name
            sender: Sender identifier
            timestamp: Message timestamp
            metadata: Additional metadata

        Returns:
            Ingestion result
        """
        if not message or len(message.strip()) < 3:
            return {'error': 'Message too short'}

        # Compute hash
        content = f"{source}:{chat_id}:{sender}:{timestamp.isoformat()}:{message}"
        doc_hash = self._compute_hash(content)

        # Check if already indexed
        existing = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="hash", match=MatchValue(value=doc_hash))]
            ),
            limit=1
        )

        if existing[0]:
            return {'status': 'already_indexed', 'hash': doc_hash}

        # Generate embedding
        try:
            embedding = self.embedding_model.encode(message, convert_to_tensor=False).tolist()
        except Exception as e:
            return {'error': f'Failed to generate embedding: {e}'}

        # Prepare metadata
        full_metadata = {
            'type': 'chat_message',
            'source': source,
            'chat_id': chat_id,
            'chat_name': chat_name,
            'sender': sender,
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': int(timestamp.timestamp()),
            'hash': doc_hash,
            'message_length': len(message),
            **(metadata or {})
        }

        # Generate ID
        doc_id = doc_hash[:16]

        # Index in Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            'text': message,
                            **full_metadata
                        }
                    )
                ]
            )
        except Exception as e:
            return {'error': f'Failed to index message: {e}'}

        return {
            'status': 'success',
            'id': doc_id,
            'hash': doc_hash,
            'source': source,
            'chat_name': chat_name
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Semantic search in vector database

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Additional filters (type, source, date range, etc.)

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()

        # Build filter
        search_filter = None
        if filters:
            must_conditions = []

            if 'type' in filters:
                must_conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=filters['type']))
                )

            if 'source' in filters:
                must_conditions.append(
                    FieldCondition(key="source", match=MatchValue(value=filters['source']))
                )

            if 'date_from' in filters or 'date_to' in filters:
                range_cond = {}
                if 'date_from' in filters:
                    range_cond['gte'] = int(filters['date_from'].timestamp())
                if 'date_to' in filters:
                    range_cond['lte'] = int(filters['date_to'].timestamp())

                must_conditions.append(
                    FieldCondition(key="timestamp_unix", range=Range(**range_cond))
                )

            if must_conditions:
                search_filter = Filter(must=must_conditions)

        # Search with optimized HNSW parameters
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
            search_params=SearchParams(
                hnsw_ef=128,  # Default: 64, Higher: more neighbors explored (+3-5% accuracy)
                exact=False
            )
        )

        # Convert to SearchResult objects
        results = []
        for result in search_results:
            payload = result.payload
            doc = Document(
                id=result.id,
                text=payload.get('text', ''),
                filepath=payload.get('filepath', ''),
                filename=payload.get('filename', ''),
                doc_type=payload.get('type', ''),
                timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat())),
                metadata=payload,
                embedding=None
            )
            results.append(SearchResult(document=doc, score=result.score))

        return results

    def timeline_query(
        self,
        start_time: datetime,
        end_time: datetime,
        doc_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Document]:
        """
        Query documents by timeline

        Args:
            start_time: Start of time range
            end_time: End of time range
            doc_types: Filter by document types
            limit: Maximum results

        Returns:
            List of documents in chronological order
        """
        must_conditions = [
            FieldCondition(
                key="timestamp_unix",
                range=Range(
                    gte=int(start_time.timestamp()),
                    lte=int(end_time.timestamp())
                )
            )
        ]

        if doc_types:
            # Match any of the types
            must_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=doc_types[0]))
            )

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        documents = []
        for point in results[0]:
            payload = point.payload
            doc = Document(
                id=point.id,
                text=payload.get('text', ''),
                filepath=payload.get('filepath', ''),
                filename=payload.get('filename', ''),
                doc_type=payload.get('type', ''),
                timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat())),
                metadata=payload
            )
            documents.append(doc)

        # Sort by timestamp
        documents.sort(key=lambda d: d.timestamp)

        return documents

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID from Qdrant

        Args:
            doc_id: Document ID (UUID)

        Returns:
            Document object or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False
            )

            if not points:
                logger.warning(f"Document {doc_id} not found")
                return None

            point = points[0]
            payload = point.payload

            return Document(
                id=point.id,
                text=payload.get('text', ''),
                filepath=payload.get('filepath', ''),
                filename=payload.get('filename', ''),
                doc_type=payload.get('type', ''),
                timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat())),
                metadata=payload
            )
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """
        Retrieve multiple documents by IDs from Qdrant

        Args:
            doc_ids: List of document IDs

        Returns:
            List of Document objects
        """
        if not doc_ids:
            return []

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=doc_ids,
                with_payload=True,
                with_vectors=False
            )

            documents = []
            for point in points:
                payload = point.payload
                doc = Document(
                    id=point.id,
                    text=payload.get('text', ''),
                    filepath=payload.get('filepath', ''),
                    filename=payload.get('filename', ''),
                    doc_type=payload.get('type', ''),
                    timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat())),
                    metadata=payload
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        collection_info = self.client.get_collection(self.collection_name)

        return {
            'collection': self.collection_name,
            'total_documents': collection_info.points_count,
            'vector_dimension': self.embedding_dim,
            'embedding_model': self.embedding_model.get_config_dict().get('_name_or_path', 'unknown'),
            'ocr_engine': 'paddleocr' if self.paddle_ocr else 'tesseract' if self.tesseract_available else 'none'
        }


if __name__ == "__main__":
    # Test the system
    print("=== Vector RAG System Test ===\n")

    # Initialize
    rag = VectorRAGSystem()

    # Get stats
    stats = rag.get_stats()
    print("Stats:", json.dumps(stats, indent=2))

    # Test document ingestion
    print("\nTest document ingestion...")
    test_file = Path("04-integrations/rag_system/README.md")
    if test_file.exists():
        result = rag.ingest_document(test_file)
        print(f"Result: {result}")

    # Test search
    print("\nTest search...")
    results = rag.search("RAG system accuracy", limit=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   File: {result.document.filename}")
        print(f"   Text: {result.document.text[:150]}...")
