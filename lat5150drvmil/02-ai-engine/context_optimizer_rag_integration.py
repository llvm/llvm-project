#!/usr/bin/env python3
"""
Context Optimizer Integration with Existing RAG/Embedding Systems

Integrates the Advanced Context Optimizer with existing high-dimensional
embedding engines instead of using separate Sentence-BERT embeddings.

This allows sharing embeddings across:
- Enhanced RAG System (enhanced_rag_system.py)
- Cognitive Memory (cognitive_memory_enhanced.py)
- Context Optimizer (advanced_context_optimizer.py)

Author: LAT5150DRVMIL AI Engine
Version: 1.0.0
"""

import logging
from typing import Optional, List, Any
import numpy as np

logger = logging.getLogger(__name__)

# Import existing embedding systems
try:
    from enhanced_rag_system import EnhancedRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("Enhanced RAG System not available")

try:
    from cognitive_memory_enhanced import CognitiveMemoryEnhanced
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False
    logger.warning("Cognitive Memory not available")

# Import context optimizer
try:
    from advanced_context_optimizer import AdvancedContextOptimizer, SemanticEmbedder
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logger.error("Advanced Context Optimizer not available!")


class UnifiedEmbeddingEngine:
    """
    Unified embedding engine that uses existing RAG/Cognitive systems
    instead of creating separate embeddings.

    Priority order:
    1. Enhanced RAG System (if available)
    2. Cognitive Memory (if available)
    3. Fallback to basic embedder
    """

    def __init__(self,
                 rag_system: Optional[EnhancedRAGSystem] = None,
                 cognitive_memory: Optional[CognitiveMemoryEnhanced] = None,
                 fallback_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize unified embedding engine

        Args:
            rag_system: Existing RAG system with embedder
            cognitive_memory: Existing cognitive memory with embedder
            fallback_model: Fallback embedding model if neither provided
        """
        self.rag_system = rag_system
        self.cognitive_memory = cognitive_memory
        self.embedder = None
        self.embedding_dim = 384  # Default

        # Priority 1: Use RAG system embedder
        if rag_system and hasattr(rag_system, 'embedder') and rag_system.embedder:
            self.embedder = rag_system.embedder
            self.embedding_dim = self._get_embedding_dim()
            logger.info(f"Using Enhanced RAG embedder ({self.embedding_dim}D)")

        # Priority 2: Use cognitive memory embedder
        elif cognitive_memory and hasattr(cognitive_memory, 'embedder') and cognitive_memory.embedder:
            self.embedder = cognitive_memory.embedder
            self.embedding_dim = self._get_embedding_dim()
            logger.info(f"Using Cognitive Memory embedder ({self.embedding_dim}D)")

        # Priority 3: Create fallback embedder
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(fallback_model)
                self.embedding_dim = self._get_embedding_dim()
                logger.info(f"Using fallback embedder: {fallback_model} ({self.embedding_dim}D)")
            except ImportError:
                logger.warning("No embedding system available - using stub")
                self.embedder = None

    def _get_embedding_dim(self) -> int:
        """Get embedding dimensionality from the embedder"""
        if self.embedder is None:
            return 384

        try:
            # Test embedding to get dimensionality
            test_embedding = self.embedder.encode("test", convert_to_numpy=True)
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return 384

    def encode(self, text: str, convert_to_numpy: bool = True) -> Any:
        """
        Encode text to embedding vector

        Args:
            text: Text to encode
            convert_to_numpy: Convert to numpy array

        Returns:
            Embedding vector (numpy array or torch tensor)
        """
        if self.embedder is None:
            # Return zero vector if no embedder
            if convert_to_numpy:
                return np.zeros(self.embedding_dim)
            else:
                return [0.0] * self.embedding_dim

        return self.embedder.encode(text, convert_to_numpy=convert_to_numpy)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts efficiently

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Array of embedding vectors
        """
        if self.embedder is None:
            return np.zeros((len(texts), self.embedding_dim))

        return self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )


class ContextOptimizerWithRAG:
    """
    Context Optimizer that integrates with existing RAG/Embedding systems

    This is a factory that creates an AdvancedContextOptimizer configured
    to use existing embedding infrastructure.
    """

    @staticmethod
    def create(
        workspace_root: str = ".",
        total_capacity: int = 200000,
        target_min_pct: float = 40.0,
        target_max_pct: float = 60.0,
        rag_system: Optional[EnhancedRAGSystem] = None,
        cognitive_memory: Optional[CognitiveMemoryEnhanced] = None,
        **kwargs
    ) -> 'AdvancedContextOptimizer':
        """
        Create AdvancedContextOptimizer integrated with existing embedding systems

        Args:
            workspace_root: Workspace root directory
            total_capacity: Total token capacity (200k for Claude)
            target_min_pct: Minimum target utilization %
            target_max_pct: Maximum target utilization %
            rag_system: Existing RAG system to share embedder with
            cognitive_memory: Existing cognitive memory to share embedder with
            **kwargs: Additional arguments for AdvancedContextOptimizer

        Returns:
            Configured AdvancedContextOptimizer instance
        """
        if not OPTIMIZER_AVAILABLE:
            raise ImportError("AdvancedContextOptimizer not available")

        # Create unified embedding engine
        unified_embedder = UnifiedEmbeddingEngine(
            rag_system=rag_system,
            cognitive_memory=cognitive_memory
        )

        # Create context optimizer
        optimizer = AdvancedContextOptimizer(
            workspace_root=workspace_root,
            total_capacity=total_capacity,
            target_min_pct=target_min_pct,
            target_max_pct=target_max_pct,
            enable_embeddings=True,
            enable_vector_db=True,
            **kwargs
        )

        # Replace the optimizer's semantic embedder with unified one
        if optimizer.semantic_embedder:
            # Inject the shared embedder
            optimizer.semantic_embedder.model = unified_embedder.embedder
            optimizer.semantic_embedder.embedding_dim = unified_embedder.embedding_dim
            logger.info(f"✓ Context optimizer using shared {unified_embedder.embedding_dim}D embeddings")

        return optimizer


def integrate_with_existing_systems(
    rag_system: Optional[EnhancedRAGSystem] = None,
    cognitive_memory: Optional[CognitiveMemoryEnhanced] = None,
    workspace_root: str = "."
) -> 'AdvancedContextOptimizer':
    """
    Convenience function to create integrated context optimizer

    Args:
        rag_system: Existing Enhanced RAG System
        cognitive_memory: Existing Cognitive Memory system
        workspace_root: Workspace root directory

    Returns:
        AdvancedContextOptimizer with shared embeddings

    Example:
        # Initialize RAG system first
        rag = EnhancedRAGSystem(embedding_model="all-mpnet-base-v2")  # 768D

        # Create context optimizer that shares RAG's embedder
        optimizer = integrate_with_existing_systems(rag_system=rag)

        # Now both use the same 768D embeddings!
        print(f"Embedding dimension: {optimizer.semantic_embedder.embedding_dim}")
    """
    return ContextOptimizerWithRAG.create(
        workspace_root=workspace_root,
        rag_system=rag_system,
        cognitive_memory=cognitive_memory
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("CONTEXT OPTIMIZER + RAG INTEGRATION DEMO")
    print("="*80 + "\n")

    # Demo 1: Using existing RAG system
    if RAG_AVAILABLE:
        print("Demo 1: Integration with Enhanced RAG System")
        print("-" * 80)

        # Initialize RAG with high-dimensional embedder
        # Options: all-MiniLM-L6-v2 (384D), all-mpnet-base-v2 (768D), gte-large (1024D)
        rag = EnhancedRAGSystem(embedding_model="all-mpnet-base-v2")  # 768D

        # Create integrated optimizer
        optimizer = integrate_with_existing_systems(rag_system=rag)

        print(f"✓ RAG embedding dimension: {rag.embedder.get_sentence_embedding_dimension()}D")
        print(f"✓ Optimizer embedding dimension: {optimizer.semantic_embedder.embedding_dim}D")
        print(f"✓ Shared embedder: {optimizer.semantic_embedder.model is rag.embedder}")
        print()

    # Demo 2: Using cognitive memory
    if COGNITIVE_AVAILABLE:
        print("Demo 2: Integration with Cognitive Memory")
        print("-" * 80)

        # Initialize cognitive memory
        cognitive = CognitiveMemoryEnhanced(enable_embeddings=True)

        # Create integrated optimizer
        optimizer = integrate_with_existing_systems(cognitive_memory=cognitive)

        print(f"✓ Cognitive embedding dimension: {cognitive.embedder.get_sentence_embedding_dimension()}D")
        print(f"✓ Optimizer embedding dimension: {optimizer.semantic_embedder.embedding_dim}D")
        print(f"✓ Shared embedder: {optimizer.semantic_embedder.model is cognitive.embedder}")
        print()

    # Demo 3: Fallback mode
    print("Demo 3: Standalone Mode (No Existing Systems)")
    print("-" * 80)

    optimizer = integrate_with_existing_systems()
    print(f"✓ Optimizer embedding dimension: {optimizer.semantic_embedder.embedding_dim}D")
    print(f"✓ Fallback embedder created")
    print()

    print("="*80)
    print("✓ All integration demos complete!")
    print("="*80)
