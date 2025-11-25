#!/usr/bin/env python3
"""
TOON Integration Utilities for AI Framework

High-level wrappers for integrating TOON compression across all AI components.

Key Benefits:
- 30-60% token savings for LLM API calls
- Smaller storage for datasets and indexes
- Faster serialization/deserialization
- Reduced bandwidth for distributed training

Usage:
    from utils.toon_integration import compress_for_llm, save_toon_json, load_toon_json

    # Compress data before sending to LLM
    prompt_data = {"context": [...], "query": "..."}
    compressed = compress_for_llm(prompt_data)

    # Save/load with automatic TOON compression
    save_toon_json("data.toon", large_dataset)
    data = load_toon_json("data.toon")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .toon_encoder import (
    json_to_toon, toon_to_json,
    ToonConfig, Delimiter,
    estimate_token_savings
)

# Default configuration for AI workloads
DEFAULT_CONFIG = ToonConfig(
    indent_size=2,
    delimiter=Delimiter.COMMA,
    use_tabular=True,
    min_tabular_rows=2,
    preserve_order=True
)


def compress_for_llm(data: Any, config: Optional[ToonConfig] = None) -> str:
    """
    Compress data for LLM prompt/context

    Use this to reduce token consumption when sending structured data to LLMs.

    Args:
        data: Python object to compress
        config: Optional TOON configuration

    Returns:
        TOON-encoded string (30-60% fewer tokens than JSON)

    Example:
        >>> context_docs = [{"title": "Doc1", "content": "..."}, ...]
        >>> toon_context = compress_for_llm(context_docs)
        >>> prompt = f"Context:\\n{toon_context}\\n\\nQuestion: {question}"
    """
    return json_to_toon(data, config or DEFAULT_CONFIG)


def decompress_from_llm(toon_str: str, config: Optional[ToonConfig] = None) -> Any:
    """
    Decompress TOON-encoded data from LLM response

    Args:
        toon_str: TOON-encoded string
        config: Optional TOON configuration

    Returns:
        Python object
    """
    return toon_to_json(toon_str, config or DEFAULT_CONFIG)


def save_toon_json(file_path: Path, data: Any, config: Optional[ToonConfig] = None) -> Dict[str, Any]:
    """
    Save data to file with TOON compression

    Args:
        file_path: Output file path (.toon extension recommended)
        data: Python object to save
        config: Optional TOON configuration

    Returns:
        Statistics dict with savings info

    Example:
        >>> stats = save_toon_json("dataset.toon", preference_pairs)
        >>> print(f"Saved {stats['toon_bytes']} bytes ({stats['savings_percent']:.1f}% smaller)")
    """
    toon_str = json_to_toon(data, config or DEFAULT_CONFIG)

    with open(file_path, 'w') as f:
        f.write(toon_str)

    # Calculate savings
    json_str = json.dumps(data, separators=(',', ':'))
    stats = {
        'json_bytes': len(json_str),
        'toon_bytes': len(toon_str),
        'bytes_saved': len(json_str) - len(toon_str),
        'savings_percent': (len(json_str) - len(toon_str)) / len(json_str) * 100 if len(json_str) > 0 else 0
    }

    return stats


def load_toon_json(file_path: Path, config: Optional[ToonConfig] = None) -> Any:
    """
    Load TOON-compressed data from file

    Args:
        file_path: Input file path
        config: Optional TOON configuration

    Returns:
        Python object

    Example:
        >>> data = load_toon_json("dataset.toon")
    """
    with open(file_path, 'r') as f:
        toon_str = f.read()

    return toon_to_json(toon_str, config or DEFAULT_CONFIG)


def compress_rag_document(doc: Dict[str, Any]) -> str:
    """
    Compress RAG document for storage

    Optimized for typical document structure:
    - metadata dict
    - content string
    - embeddings array (if present)

    Args:
        doc: Document dict with keys like {title, content, metadata, embeddings}

    Returns:
        TOON-encoded document string
    """
    return compress_for_llm(doc)


def compress_dpo_dataset(preference_pairs: List[Dict[str, Any]]) -> str:
    """
    Compress DPO preference pairs for storage

    Typical structure:
    [
        {"query": "...", "chosen": "...", "rejected": "...", "metadata": {...}},
        ...
    ]

    TOON tabular format provides 40-60% savings on uniform preference pairs.

    Args:
        preference_pairs: List of DPO preference pair dicts

    Returns:
        TOON-encoded dataset string
    """
    return compress_for_llm(preference_pairs)


def compress_cluster_metadata(cluster_info: Dict[str, Any]) -> str:
    """
    Compress multi-GPU cluster metadata for network transmission

    Reduces bandwidth when exchanging cluster info between nodes.

    Args:
        cluster_info: Cluster metadata dict

    Returns:
        TOON-encoded cluster info
    """
    return compress_for_llm(cluster_info)


def compress_telegram_index(index: Dict[str, Any]) -> str:
    """
    Compress Telegram scraper security index

    Typical index structure:
    {
        "cves": {cve_id: {...}, ...},
        "documents": {doc_id: {...}, ...},
        "files": {hash: {...}, ...}
    }

    Args:
        index: Security index dict

    Returns:
        TOON-encoded index
    """
    return compress_for_llm(index)


class TOONSerializer:
    """
    Drop-in replacement for json module with TOON compression

    Usage:
        import toon_integration
        toon_json = toon_integration.TOONSerializer()

        # Use like json module
        data_str = toon_json.dumps(data)
        data = toon_json.loads(data_str)

        # Save/load files
        with open('data.toon', 'w') as f:
            toon_json.dump(data, f)
    """

    def __init__(self, config: Optional[ToonConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def dumps(self, obj: Any) -> str:
        """Serialize object to TOON string"""
        return json_to_toon(obj, self.config)

    def loads(self, s: str) -> Any:
        """Deserialize TOON string to object"""
        return toon_to_json(s, self.config)

    def dump(self, obj: Any, fp):
        """Serialize object to file-like object"""
        toon_str = self.dumps(obj)
        fp.write(toon_str)

    def load(self, fp) -> Any:
        """Deserialize file-like object to object"""
        toon_str = fp.read()
        return self.loads(toon_str)


# Convenience instance
toon_json = TOONSerializer()


if __name__ == "__main__":
    print("=" * 80)
    print("TOON Integration Utilities - Demo")
    print("=" * 80)

    # Demo 1: Compress for LLM
    print("\n1. Compress RAG context for LLM:")
    context_docs = [
        {"id": 1, "title": "CVE-2024-1234", "severity": "Critical", "score": 9.8},
        {"id": 2, "title": "CVE-2024-5678", "severity": "High", "score": 7.5},
        {"id": 3, "title": "CVE-2024-9012", "severity": "Medium", "score": 5.3}
    ]

    json_version = json.dumps(context_docs)
    toon_version = compress_for_llm(context_docs)

    print(f"JSON ({len(json_version)} chars):")
    print(json_version)
    print(f"\nTOON ({len(toon_version)} chars, {(1 - len(toon_version)/len(json_version))*100:.1f}% smaller):")
    print(toon_version)

    # Demo 2: Save/Load with compression
    print("\n2. Save/Load DPO Dataset:")
    dpo_data = [
        {"query": f"Question {i}", "chosen": f"Good answer {i}", "rejected": f"Bad answer {i}"}
        for i in range(10)
    ]

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toon', delete=False) as f:
        stats = save_toon_json(f.name, dpo_data)
        temp_file = f.name

    print(f"Saved to {temp_file}")
    print(f"  JSON would be: {stats['json_bytes']} bytes")
    print(f"  TOON is: {stats['toon_bytes']} bytes")
    print(f"  Savings: {stats['savings_percent']:.1f}%")

    loaded = load_toon_json(temp_file)
    print(f"  Round-trip match: {loaded == dpo_data}")

    import os
    os.unlink(temp_file)

    # Demo 3: Token savings estimation
    print("\n3. Token Savings for Large Dataset:")
    large_dataset = [
        {"id": i, "user": f"user{i}", "score": i * 1.5, "active": i % 2 == 0}
        for i in range(100)
    ]

    savings = estimate_token_savings(large_dataset)
    print(f"100 uniform records:")
    print(f"  JSON tokens: {savings['json_tokens']}")
    print(f"  TOON tokens: {savings['toon_tokens']}")
    print(f"  Savings: {savings['tokens_saved']} tokens ({savings['savings_percent']:.1f}%)")

    print("\n" + "=" * 80)
