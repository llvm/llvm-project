#!/usr/bin/env python3
"""
Comprehensive Test Suite for TOON Encoder/Decoder

Tests the TOON v1.4 implementation for:
- Basic encoding/decoding
- Tabular format optimization
- Nested object handling
- Round-trip integrity
- Token savings verification
- Edge cases and error handling
"""

import pytest
import json
from utils.toon_encoder import (
    json_to_toon, toon_to_json,
    ToonEncoder, ToonDecoder,
    ToonConfig, Delimiter,
    estimate_token_savings,
    EncodingError, DecodingError
)
from utils.toon_integration import (
    compress_for_llm, decompress_from_llm,
    save_toon_json, load_toon_json,
    compress_rag_document, compress_dpo_dataset,
    toon_json
)


class TestBasicEncoding:
    """Test basic TOON encoding functionality"""

    def test_simple_object(self):
        """Test simple object encoding"""
        data = {"name": "Alice", "age": 30, "active": True}
        toon = json_to_toon(data)
        assert "name: Alice" in toon
        assert "age: 30" in toon
        assert "active: true" in toon

    def test_simple_array(self):
        """Test simple array encoding"""
        data = [1, 2, 3, 4, 5]
        toon = json_to_toon(data)
        assert "[5]:" in toon
        assert "1,2,3,4,5" in toon

    def test_nested_object(self):
        """Test nested object encoding"""
        data = {
            "user": {
                "name": "Alice",
                "email": "alice@example.com"
            },
            "metadata": {
                "created": "2025-11-09",
                "active": True
            }
        }
        toon = json_to_toon(data)
        assert "user:" in toon
        assert "name: Alice" in toon
        assert "metadata:" in toon

    def test_mixed_types(self):
        """Test various data types"""
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3]
        }
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data


class TestTabularFormat:
    """Test TOON tabular format optimization"""

    def test_uniform_objects(self):
        """Test tabular format for uniform objects"""
        data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.0},
            {"id": 3, "name": "Charlie", "score": 92.3}
        ]
        toon = json_to_toon(data)

        # Should use tabular format
        assert "[3]{id,name,score}:" in toon
        assert "1,Alice,95.5" in toon
        assert "2,Bob,87" in toon
        assert "3,Charlie,92.3" in toon

    def test_large_uniform_dataset(self):
        """Test tabular format with large dataset"""
        data = [
            {"id": i, "user": f"user{i}", "score": i * 1.5, "active": i % 2 == 0}
            for i in range(100)
        ]

        savings = estimate_token_savings(data)

        # Should achieve significant savings
        assert savings['savings_percent'] > 50, "Should save >50% tokens on uniform data"

        # Verify round-trip
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_dpo_preference_pairs(self):
        """Test DPO dataset compression"""
        dpo_data = [
            {
                "query": f"Question {i}",
                "chosen": f"Good answer {i}",
                "rejected": f"Bad answer {i}",
                "score": 0.75 + (i % 10) * 0.02
            }
            for i in range(50)
        ]

        toon = compress_dpo_dataset(dpo_data)
        result = toon_to_json(toon)
        assert result == dpo_data


class TestRoundTrip:
    """Test round-trip encoding/decoding integrity"""

    def test_cve_records(self):
        """Test with realistic CVE data"""
        data = [
            {
                "cve_id": "CVE-2024-1234",
                "severity": "Critical",
                "score": 9.8,
                "description": "Remote code execution vulnerability",
                "published": "2024-01-15"
            },
            {
                "cve_id": "CVE-2024-5678",
                "severity": "High",
                "score": 7.5,
                "description": "SQL injection vulnerability",
                "published": "2024-02-20"
            }
        ]

        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_rag_document(self):
        """Test RAG document compression"""
        doc = {
            "id": "doc123",
            "title": "Security Advisory",
            "content": "This is a security advisory about CVE-2024-1234...",
            "metadata": {
                "source": "NVD",
                "severity": "Critical",
                "tags": ["rce", "network", "authentication"]
            }
        }

        toon = compress_rag_document(doc)
        result = toon_to_json(toon)
        assert result == doc

    def test_complex_nested_structure(self):
        """Test complex nested structure"""
        data = {
            "users": [
                {
                    "id": 1,
                    "name": "Alice",
                    "roles": ["admin", "developer"],
                    "metadata": {
                        "last_login": "2025-11-09T10:00:00Z",
                        "active": True
                    }
                },
                {
                    "id": 2,
                    "name": "Bob",
                    "roles": ["user"],
                    "metadata": {
                        "last_login": "2025-11-08T15:30:00Z",
                        "active": False
                    }
                }
            ],
            "permissions": {
                "admin": ["read", "write", "delete"],
                "user": ["read"]
            }
        }

        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data


class TestTokenSavings:
    """Test token savings calculations"""

    def test_token_savings_estimation(self):
        """Test token savings estimation accuracy"""
        data = [
            {"id": i, "name": f"User{i}", "score": i * 1.5}
            for i in range(100)
        ]

        savings = estimate_token_savings(data)

        assert 'json_tokens' in savings
        assert 'toon_tokens' in savings
        assert 'tokens_saved' in savings
        assert 'savings_percent' in savings

        # Verify math
        assert savings['tokens_saved'] == savings['json_tokens'] - savings['toon_tokens']
        assert abs(savings['savings_percent'] -
                   (savings['tokens_saved'] / savings['json_tokens'] * 100)) < 0.01

    def test_actual_vs_estimated_savings(self):
        """Compare actual string length to token estimation"""
        data = [
            {"id": i, "value": i * 2, "flag": i % 2 == 0}
            for i in range(50)
        ]

        json_str = json.dumps(data, separators=(',', ':'))
        toon_str = json_to_toon(data)

        actual_savings_pct = (len(json_str) - len(toon_str)) / len(json_str) * 100

        savings = estimate_token_savings(data)

        # Token estimate should be close to character count savings
        assert abs(actual_savings_pct - savings['savings_percent']) < 15


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_object(self):
        """Test empty object"""
        data = {}
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_empty_array(self):
        """Test empty array"""
        data = []
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_none_values(self):
        """Test None/null values"""
        data = {"key1": None, "key2": "value", "key3": None}
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_special_characters(self):
        """Test special characters in strings"""
        data = {
            "newline": "line1\nline2",
            "tab": "col1\tcol2",
            "quote": 'He said "hello"',
            "backslash": "path\\to\\file"
        }
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_unicode_characters(self):
        """Test Unicode characters"""
        data = {
            "emoji": "Hello 👋 World 🌍",
            "chinese": "你好世界",
            "arabic": "مرحبا بالعالم",
            "math": "∑ ∫ ∞ ≈"
        }
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data

    def test_float_precision(self):
        """Test floating point precision"""
        data = {
            "pi": 3.14159,
            "e": 2.71828,
            "small": 0.00001,
            "large": 123456.789
        }
        toon = json_to_toon(data)
        result = toon_to_json(toon)

        # Check close equality for floats
        assert abs(result["pi"] - data["pi"]) < 1e-10
        assert abs(result["e"] - data["e"]) < 1e-10

    def test_large_numbers(self):
        """Test large integer values"""
        data = {
            "max_int": 2**53 - 1,  # JavaScript safe integer
            "large": 1234567890123456,
            "negative": -9876543210
        }
        toon = json_to_toon(data)
        result = toon_to_json(toon)
        assert result == data


class TestFileOperations:
    """Test file save/load operations"""

    def test_save_and_load_toon_file(self, tmp_path):
        """Test saving and loading TOON files"""
        data = [
            {"id": i, "name": f"Item{i}", "value": i * 2.5}
            for i in range(20)
        ]

        file_path = tmp_path / "test_data.toon"

        # Save
        stats = save_toon_json(file_path, data)

        assert stats['toon_bytes'] > 0
        assert stats['json_bytes'] > stats['toon_bytes']
        assert stats['savings_percent'] > 0

        # Load
        loaded = load_toon_json(file_path)
        assert loaded == data

    def test_toon_serializer_interface(self, tmp_path):
        """Test TOONSerializer as drop-in json replacement"""
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

        # Test dumps/loads
        toon_str = toon_json.dumps(data)
        result = toon_json.loads(toon_str)
        assert result == data

        # Test dump/load
        file_path = tmp_path / "test.toon"
        with open(file_path, 'w') as f:
            toon_json.dump(data, f)

        with open(file_path, 'r') as f:
            loaded = toon_json.load(f)

        assert loaded == data


class TestConfiguration:
    """Test TOON configuration options"""

    def test_delimiter_options(self):
        """Test different delimiter configurations"""
        data = {"a": 1, "b": 2, "c": 3}

        # Comma delimiter (default)
        config_comma = ToonConfig(delimiter=Delimiter.COMMA)
        toon_comma = json_to_toon(data, config_comma)

        # Tab delimiter
        config_tab = ToonConfig(delimiter=Delimiter.TAB)
        toon_tab = json_to_toon(data, config_tab)

        # Both should decode correctly
        assert toon_to_json(toon_comma, config_comma) == data
        assert toon_to_json(toon_tab, config_tab) == data

    def test_indent_size(self):
        """Test different indentation sizes"""
        data = {"parent": {"child": "value"}}

        config_2 = ToonConfig(indent_size=2)
        toon_2 = json_to_toon(data, config_2)

        config_4 = ToonConfig(indent_size=4)
        toon_4 = json_to_toon(data, config_4)

        # Both should decode correctly
        assert toon_to_json(toon_2, config_2) == data
        assert toon_to_json(toon_4, config_4) == data

    def test_tabular_toggle(self):
        """Test disabling tabular format"""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]

        # With tabular (default)
        config_tabular = ToonConfig(use_tabular=True)
        toon_tabular = json_to_toon(data, config_tabular)
        assert "[2]{id,name}:" in toon_tabular

        # Without tabular
        config_no_tabular = ToonConfig(use_tabular=False)
        toon_no_tabular = json_to_toon(data, config_no_tabular)
        assert "[2]{id,name}:" not in toon_no_tabular

        # Both should decode correctly
        assert toon_to_json(toon_tabular, config_tabular) == data
        assert toon_to_json(toon_no_tabular, config_no_tabular) == data


def test_compress_for_llm():
    """Test high-level LLM compression wrapper"""
    context = [
        {"doc_id": 1, "title": "Doc1", "relevance": 0.95},
        {"doc_id": 2, "title": "Doc2", "relevance": 0.87},
        {"doc_id": 3, "title": "Doc3", "relevance": 0.76}
    ]

    compressed = compress_for_llm(context)
    decompressed = decompress_from_llm(compressed)

    assert decompressed == context


def test_readme_examples():
    """Test all examples from README_TOON.md work correctly"""

    # Example 1: Simple tabular
    data = [
        {"id": 1, "name": "Alice", "score": 95.5},
        {"id": 2, "name": "Bob", "score": 87.0},
        {"id": 3, "name": "Charlie", "score": 92.3}
    ]

    toon = json_to_toon(data)
    assert toon_to_json(toon) == data

    # Example 2: CVE records
    cve_data = [
        {"cve_id": "CVE-2024-1234", "severity": "Critical", "score": 9.8},
        {"cve_id": "CVE-2024-5678", "severity": "High", "score": 7.5}
    ]

    savings = estimate_token_savings(cve_data)
    assert savings['savings_percent'] > 30  # Should save >30% tokens


if __name__ == '__main__':
    # Run tests
    print("=" * 80)
    print("TOON Encoder/Decoder Test Suite")
    print("=" * 80)

    pytest.main([__file__, '-v', '--tb=short'])
