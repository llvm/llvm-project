"""
Deep-Thinking RAG Components

Implements advanced RAG with:
- Cross-encoder reranking for precision
- Multi-phase reasoning (Plan-Retrieve-Reflect-Critique-Synthesis)
- Adaptive retrieval strategies
- Policy-based control flow
"""

from .cross_encoder_reranker import CrossEncoderReranker

__all__ = ['CrossEncoderReranker']
