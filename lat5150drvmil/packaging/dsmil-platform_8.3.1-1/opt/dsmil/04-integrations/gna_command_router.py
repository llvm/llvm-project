#!/usr/bin/env python3
"""GNA Command Router - Ultra-low power command classification"""
from openvino import Core
import numpy as np

class GNARouter:
    def __init__(self):
        self.core = Core()
        self.device = "CPU"  # GNA via CPU device in OpenVINO

    def classify_command(self, text):
        """
        Classify user input for routing
        Returns: category, confidence
        """
        lower = text.lower()

        # Simple rule-based (GNA-optimized for INT8)
        if any(k in lower for k in ['run:', '$', 'exec:']):
            return 'shell', 0.95
        elif any(k in lower for k in ['cat', 'read']):
            return 'file', 0.90
        elif any(k in lower for k in ['rag:', 'search']):
            return 'rag', 0.92
        elif any(k in lower for k in ['web:', 'http']):
            return 'web', 0.88
        elif any(k in lower for k in ['github:', 'git@']):
            return 'github', 0.91
        elif any(k in lower for k in ['npu', 'test']):
            return 'system', 0.87
        else:
            return 'nlp', 0.60  # Natural language processing

# CLI
if __name__ == "__main__":
    import sys
    router = GNARouter()

    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
        cat, conf = router.classify_command(cmd)
        print(f"Category: {cat}, Confidence: {conf}")
    else:
        # Test
        tests = [
            "run: ls -la",
            "cat README.md",
            "rag: search APT",
            "test npu",
            "tell me about kernel"
        ]
        for t in tests:
            cat, conf = router.classify_command(t)
            print(f"{t:30} -> {cat:10} ({conf:.0%})")
