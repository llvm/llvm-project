"""
AI Framework Utilities

Core utilities for token optimization and data compression.
"""

from .toon_encoder import (
    json_to_toon,
    toon_to_json,
    ToonEncoder,
    ToonDecoder,
    ToonConfig,
    Delimiter,
    estimate_token_savings
)

from .toon_integration import (
    compress_for_llm,
    decompress_from_llm,
    save_toon_json,
    load_toon_json,
    compress_rag_document,
    compress_dpo_dataset,
    compress_cluster_metadata,
    compress_telegram_index,
    TOONSerializer,
    toon_json
)

__all__ = [
    # TOON encoder/decoder
    'json_to_toon',
    'toon_to_json',
    'ToonEncoder',
    'ToonDecoder',
    'ToonConfig',
    'Delimiter',
    'estimate_token_savings',
    # TOON integration
    'compress_for_llm',
    'decompress_from_llm',
    'save_toon_json',
    'load_toon_json',
    'compress_rag_document',
    'compress_dpo_dataset',
    'compress_cluster_metadata',
    'compress_telegram_index',
    'TOONSerializer',
    'toon_json',
]
