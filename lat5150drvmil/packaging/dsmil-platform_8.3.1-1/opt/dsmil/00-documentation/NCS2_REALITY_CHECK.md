# NCS2 Reality Check - Actual Performance

## Theoretical vs Actual

SPEC SHEET:
- 1 TOPS (INT8)
- USB 3.0 interface
- 5 Gbps theoretical

REALITY:
- USB bottleneck: ~400 MB/s actual
- Latency: USB overhead ~2-5ms
- Best for: Small models, embeddings, not main inference

## Comparison to Your Hardware

NPU 3720: 34 TOPS
NCS2: 1 TOPS
RATIO: 34:1 (NPU is 34x more powerful)

## Verdict

NCS2 WORTH IT FOR:
- Embeddings only (offload from CPU)
- Background monitoring (dedicated, low power)
- NOT for main inference (NPU is far superior)

RECOMMENDATION:
- Use NPU for AI inference (34x faster)
- Use NCS2 for embeddings/monitoring only
- Focus optimization on NPU + P-cores

TOKEN COST TO INTEGRATE NCS2: 15K
BENEFIT: Marginal (+5% vs NPU-only)
PRIORITY: LOW

FOCUS INSTEAD:
- NPU optimization (34 TOPS)
- AVX-512 on P-cores (2-4x speedup)
- Proper quantization (INT4/INT8)

COST: 30K tokens
BENEFIT: 3-5x performance gain
PRIORITY: HIGH
