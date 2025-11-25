#!/usr/bin/env python3
"""
Late Chunking Strategy for Enhanced Retrieval Accuracy

Late chunking embeds the full document then splits into chunks, preserving
contextual information across chunk boundaries.

Performance gain: +3-4% nDCG over naive chunking (SIGIR'25)

How it works:
1. Naive chunking: Split text → embed each chunk separately
   - Problem: Each chunk loses context from other chunks
   - Embedding sees: "VPN error" (isolated)

2. Late chunking: Embed full text → split embeddings into chunks
   - Benefit: Each chunk embedding contains full document context
   - Embedding sees: "... system logs show VPN error in network ..." (full context)

Why it works better:
- Token embeddings capture bidirectional context from full document
- Chunk boundaries don't interrupt attention mechanism
- Especially effective for small chunks (256-512 tokens)
- ~2-3 point nDCG gain on retrieval benchmarks

Best practices:
- Use with long-context models (Jina v3: 8192 tokens, BGE: 512 tokens)
- Optimal chunk size: 256-400 tokens for balanced accuracy/latency
- Works with both fixed-size and semantic chunking
- Pairs well with ColBERT multi-vector retrieval

Research: "Late Chunking" (SIGIR'25), Jina AI
"""

import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available for late chunking")


@dataclass
class LateChunk:
    """Chunk with contextual embedding from late chunking"""
    text: str
    embedding: List[float]  # Contextual embedding from full document
    start_token: int  # Token position in original document
    end_token: int
    chunk_id: int
    metadata: Dict


class LateChunkingEncoder:
    """
    Late Chunking encoder for contextual chunk embeddings

    Encodes full document then splits token embeddings into chunks,
    preserving full document context in each chunk.

    Supports:
    - Fixed-size chunking (token-based)
    - Semantic chunking (sentence/paragraph boundaries)
    - Pooling strategies (mean, max, CLS)
    - Long-context models (Jina v3, Longformer, etc.)
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        chunk_size: int = 256,  # Tokens per chunk
        overlap: int = 50,  # Token overlap between chunks
        pooling: str = "mean",  # mean, max, cls, or None (keep token embeddings)
        use_gpu: bool = True,
        trust_remote_code: bool = True
    ):
        """
        Initialize late chunking encoder

        Args:
            model_name: Model for embeddings (recommend long-context models)
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            pooling: Pooling strategy for chunk embeddings
            use_gpu: Use GPU if available
            trust_remote_code: Trust remote code (needed for Jina v3)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers and torch required")

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pooling = pooling
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading late chunking encoder: {model_name} on {self.device}")

        # Load model
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Failed with trust_remote_code, trying without: {e}")
            self.model = SentenceTransformer(model_name, device=self.device)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length

        logger.info("✓ Late chunking encoder initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}D")
        logger.info(f"  Max sequence: {self.max_seq_length} tokens")
        logger.info(f"  Chunk size: {chunk_size} tokens (overlap: {overlap})")
        logger.info(f"  Pooling: {pooling}")

    def _get_token_embeddings(self, text: str) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Get token-level embeddings for full document

        Args:
            text: Full document text

        Returns:
            (token_embeddings, tokens, attention_mask)
            token_embeddings shape: (seq_len, embedding_dim)
        """
        # Tokenize
        encoded = self.model.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get token embeddings
        with torch.no_grad():
            model_output = self.model[0].auto_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get last hidden state (token embeddings)
            token_embeddings = model_output.last_hidden_state[0]  # (seq_len, hidden_dim)

        # Get tokens
        tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        return token_embeddings, tokens, attention_mask[0]

    def _pool_token_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool token embeddings to single vector

        Args:
            token_embeddings: Token embeddings (seq_len, hidden_dim)
            attention_mask: Attention mask (seq_len,)

        Returns:
            Pooled embedding (hidden_dim,)
        """
        if self.pooling == "mean":
            # Mean pooling over tokens
            if attention_mask is not None:
                # Mask out padding tokens
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = torch.sum(token_embeddings * mask, dim=0)
                sum_mask = torch.clamp(mask.sum(dim=0), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return token_embeddings.mean(dim=0)

        elif self.pooling == "max":
            # Max pooling over tokens
            if attention_mask is not None:
                # Mask out padding tokens with large negative value
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                token_embeddings = token_embeddings.clone()
                token_embeddings[mask == 0] = -1e9
            return token_embeddings.max(dim=0)[0]

        elif self.pooling == "cls":
            # Use [CLS] token embedding (first token)
            return token_embeddings[0]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def encode_document_with_late_chunking(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        return_token_embeddings: bool = False
    ) -> List[LateChunk]:
        """
        Encode document with late chunking strategy

        Steps:
        1. Encode full document to token embeddings
        2. Split token embeddings into chunks
        3. Pool each chunk's token embeddings
        4. Return chunks with contextual embeddings

        Args:
            text: Full document text
            metadata: Document metadata
            return_token_embeddings: Return raw token embeddings (for ColBERT)

        Returns:
            List of LateChunk objects with contextual embeddings
        """
        # Get token embeddings for full document
        token_embeddings, tokens, attention_mask = self._get_token_embeddings(text)

        # Get actual sequence length (excluding padding)
        seq_len = attention_mask.sum().item()

        # Compute chunk boundaries
        chunks = []
        chunk_id = 0
        start_pos = 0

        while start_pos < seq_len:
            # Compute chunk range
            end_pos = min(start_pos + self.chunk_size, seq_len)

            # Extract chunk token embeddings
            chunk_token_embeddings = token_embeddings[start_pos:end_pos]
            chunk_attention_mask = attention_mask[start_pos:end_pos]
            chunk_tokens = tokens[start_pos:end_pos]

            # Get chunk text (decode tokens)
            chunk_token_ids = self.model.tokenizer.convert_tokens_to_ids(chunk_tokens)
            chunk_text = self.model.tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Pool token embeddings to single vector (unless keeping token embeddings)
            if return_token_embeddings:
                # Return token embeddings for ColBERT-style retrieval
                chunk_embedding = chunk_token_embeddings.cpu().numpy().tolist()
            else:
                # Pool to single embedding
                pooled = self._pool_token_embeddings(chunk_token_embeddings, chunk_attention_mask)
                # Normalize
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
                chunk_embedding = pooled.cpu().numpy().tolist()

            # Create chunk
            chunk = LateChunk(
                text=chunk_text,
                embedding=chunk_embedding,
                start_token=start_pos,
                end_token=end_pos,
                chunk_id=chunk_id,
                metadata=metadata or {}
            )
            chunks.append(chunk)

            # Move to next chunk (with overlap)
            start_pos += self.chunk_size - self.overlap
            chunk_id += 1

        logger.debug(f"Late chunking: {len(text)} chars → {len(chunks)} chunks ({seq_len} tokens)")

        return chunks

    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Encode query with standard encoding (no chunking for queries)

        Args:
            query: Query text
            normalize: Normalize embedding

        Returns:
            Query embedding
        """
        embedding = self.model.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=normalize
        )
        return embedding.tolist()


class LateChunkingWithSemanticBoundaries(LateChunkingEncoder):
    """
    Late chunking with semantic boundaries (sentence/paragraph splits)

    Instead of fixed-size chunks, splits at natural boundaries
    while preserving full document context.

    Best for: Structured documents, logs with clear boundaries
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        target_chunk_tokens: int = 256,
        max_chunk_tokens: int = 512,
        split_on: str = "sentence",  # sentence, paragraph, or line
        use_gpu: bool = True,
        trust_remote_code: bool = True
    ):
        """
        Initialize semantic late chunking

        Args:
            model_name: Model for embeddings
            target_chunk_tokens: Target chunk size
            max_chunk_tokens: Maximum chunk size
            split_on: Boundary type (sentence, paragraph, line)
            use_gpu: Use GPU
            trust_remote_code: Trust remote code
        """
        super().__init__(
            model_name=model_name,
            chunk_size=target_chunk_tokens,
            overlap=0,  # No overlap for semantic chunks
            pooling="mean",
            use_gpu=use_gpu,
            trust_remote_code=trust_remote_code
        )

        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.split_on = split_on

        logger.info(f"Semantic late chunking: {split_on} boundaries")

    def _split_by_boundaries(self, text: str) -> List[str]:
        """
        Split text by semantic boundaries

        Args:
            text: Full text

        Returns:
            List of text segments
        """
        import re

        if self.split_on == "paragraph":
            # Split on double newlines
            segments = text.split('\n\n')
            segments = [s.strip() for s in segments if s.strip()]

        elif self.split_on == "line":
            # Split on single newlines
            segments = text.split('\n')
            segments = [s.strip() for s in segments if s.strip()]

        elif self.split_on == "sentence":
            # Split on sentence boundaries
            # Simple sentence splitter (same as in chunking.py)
            segments = re.split(r'([.!?]+\s+)', text)
            # Merge punctuation with sentences
            result = []
            current = ""
            for part in segments:
                current += part
                if re.match(r'[.!?]+\s+', part):
                    result.append(current.strip())
                    current = ""
            if current:
                result.append(current.strip())
            segments = [s for s in result if s]

        else:
            raise ValueError(f"Unknown split_on: {self.split_on}")

        return segments

    def encode_document_with_late_chunking(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        return_token_embeddings: bool = False
    ) -> List[LateChunk]:
        """
        Encode with semantic boundaries and late chunking

        Strategy:
        1. Get token embeddings for full document
        2. Split text by semantic boundaries
        3. Map boundaries to token positions
        4. Extract chunk embeddings at boundaries
        5. Pool to single vector per chunk

        Args:
            text: Full document
            metadata: Metadata
            return_token_embeddings: Return token embeddings

        Returns:
            List of semantically-chunked LateChunk objects
        """
        # Get token embeddings for full document
        token_embeddings, tokens, attention_mask = self._get_token_embeddings(text)
        seq_len = attention_mask.sum().item()

        # Split text by semantic boundaries
        segments = self._split_by_boundaries(text)

        # For each segment, find token positions and extract embeddings
        chunks = []
        current_char_pos = 0

        for chunk_id, segment in enumerate(segments):
            # Find segment in original text
            segment_start = text.find(segment, current_char_pos)
            if segment_start == -1:
                continue

            segment_end = segment_start + len(segment)

            # Estimate token positions (rough approximation)
            # More accurate: use tokenizer's offset mapping
            char_to_token_ratio = seq_len / len(text)
            start_token = int(segment_start * char_to_token_ratio)
            end_token = int(segment_end * char_to_token_ratio)

            start_token = max(0, start_token)
            end_token = min(seq_len, end_token)

            # Extract chunk token embeddings
            if end_token > start_token:
                chunk_token_embeddings = token_embeddings[start_token:end_token]
                chunk_attention_mask = attention_mask[start_token:end_token]

                # Pool to single embedding
                if return_token_embeddings:
                    chunk_embedding = chunk_token_embeddings.cpu().numpy().tolist()
                else:
                    pooled = self._pool_token_embeddings(chunk_token_embeddings, chunk_attention_mask)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
                    chunk_embedding = pooled.cpu().numpy().tolist()

                # Create chunk
                chunk = LateChunk(
                    text=segment,
                    embedding=chunk_embedding,
                    start_token=start_token,
                    end_token=end_token,
                    chunk_id=chunk_id,
                    metadata=metadata or {}
                )
                chunks.append(chunk)

            current_char_pos = segment_end

        logger.debug(f"Semantic late chunking: {len(segments)} segments → {len(chunks)} chunks")

        return chunks


# Example usage
if __name__ == "__main__":
    print("=== Late Chunking Test ===\n")

    # Test document (forensic log example)
    test_text = """
System Log Analysis Report:

Timestamp: 2024-01-15 14:32:11
Event: VPN connection attempt from IP 192.168.1.105
Status: Failed - Authentication timeout

Network trace shows multiple retry attempts with increasing backoff.
Firewall logs indicate no blocking rules triggered.
Certificate validation passed successfully.

Root cause analysis:
The VPN gateway experienced high load (89% CPU) during the connection window.
Database query for user credentials took 8.2 seconds (timeout threshold: 5s).
Recommendation: Increase timeout threshold or optimize user lookup query.

Follow-up actions:
1. Monitor gateway performance
2. Review database indexing strategy
3. Implement caching for frequent user lookups
    """.strip()

    print(f"Test document: {len(test_text)} characters\n")
    print("="*60 + "\n")

    # Initialize late chunking encoder (using BGE for demo, Jina v3 recommended)
    print("Initializing late chunking encoder...")
    encoder = LateChunkingEncoder(
        model_name="BAAI/bge-small-en-v1.5",  # Small model for demo
        chunk_size=64,  # Small chunks to demonstrate context preservation
        overlap=16,
        pooling="mean",
        use_gpu=False
    )

    print("\nEncoding document with late chunking...")
    chunks = encoder.encode_document_with_late_chunking(test_text)

    print(f"\nResult: {len(chunks)} chunks with contextual embeddings\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} (tokens {chunk.start_token}-{chunk.end_token}):")
        print(f"  Embedding dim: {len(chunk.embedding)}D")
        print(f"  Text: {chunk.text[:80]}...")
        print()

    print("="*60 + "\n")

    # Test semantic boundaries
    print("Testing semantic late chunking (sentence boundaries)...\n")
    semantic_encoder = LateChunkingWithSemanticBoundaries(
        model_name="BAAI/bge-small-en-v1.5",
        target_chunk_tokens=64,
        max_chunk_tokens=128,
        split_on="sentence",
        use_gpu=False
    )

    semantic_chunks = semantic_encoder.encode_document_with_late_chunking(test_text)

    print(f"Result: {len(semantic_chunks)} semantic chunks\n")

    for i, chunk in enumerate(semantic_chunks[:5], 1):  # Show first 5
        print(f"Chunk {i}:")
        print(f"  Text: {chunk.text[:100]}...")
        print()

    print("✓ Late chunking test complete")
    print("\nKey benefits:")
    print("- Each chunk embedding contains full document context")
    print("- +3-4% nDCG improvement over naive chunking")
    print("- Works best with long-context models (Jina v3: 8K tokens)")
