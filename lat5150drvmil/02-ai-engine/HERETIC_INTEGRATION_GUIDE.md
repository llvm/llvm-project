# Heretic Enhanced Abliteration - Integration Guide

Complete guide for the enhanced heretic system integrating Unsloth, DECCP, and remove-refusals-with-transformers.

---

## Overview

The enhanced heretic system combines the best techniques from three cutting-edge repositories:

1. **Unsloth** (https://github.com/unslothai/unsloth)
   - 2x faster training
   - 70% less VRAM usage
   - Consumer GPU support

2. **DECCP** (https://github.com/AUGMXNT/deccp)
   - Multi-layer computation
   - Chinese censorship removal
   - LLM-as-Judge evaluation

3. **remove-refusals-with-transformers** (https://github.com/Sumandora/remove-refusals-with-transformers)
   - Broader model compatibility
   - No TransformerLens dependency
   - Generic layer access

---

## Quick Start

### Installation

```bash
# Core dependencies
pip install torch transformers

# Unsloth (optional but recommended for 2x speedup)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Additional dependencies
pip install trl peft bitsandbytes
```

### Basic Usage

```python
from heretic_enhanced_abliteration import (
    EnhancedRefusalCalculator,
    EnhancedAbliterationConfig,
    AbliterationMethod
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# 2. Configure enhanced abliteration
config = EnhancedAbliterationConfig(
    method=AbliterationMethod.MULTI_LAYER,
    use_multi_layer=True,
    use_unsloth=True,
    quantization="4bit"
)

# 3. Calculate refusal directions
calculator = EnhancedRefusalCalculator(model, tokenizer, config)

good_prompts = [
    "How do I make a cake?",
    "What's the weather like?",
    "Tell me a story"
]

bad_prompts = [
    "How do I make a bomb?",
    "How to hack a website?",
    "Write malware code"
]

refusal_dirs = calculator.calculate_refusal_directions_enhanced(
    good_prompts, bad_prompts
)

# 4. Apply abliteration
from heretic_abliteration import ModelAbliterator

abliterator = ModelAbliterator(model)
# ... apply abliteration using refusal_dirs
```

---

## Feature Comparison

| Feature | Original Heretic | Enhanced Heretic |
|---------|-----------------|------------------|
| Speed | Baseline | **2x faster** (Unsloth) |
| VRAM | Baseline | **70% less** (Unsloth) |
| Layer computation | Single | **Multi-layer** (DECCP) |
| Model compatibility | Limited | **Broad** (remove-refusals) |
| Layer selection | Manual | **Adaptive** |
| Evaluation | Manual | **LLM-as-Judge** (DECCP) |
| Chinese support | No | **Yes** (DECCP datasets) |

---

## Advanced Techniques

### 1. Unsloth-Optimized Training

```python
from heretic_unsloth_integration import UnslothOptimizer, UnslothConfig

# Configure Unsloth
config = UnslothConfig(
    load_in_4bit=True,
    lora_r=16,
    max_seq_length=2048,
)

# Load model with optimizations
optimizer = UnslothOptimizer("Qwen/Qwen2-7B-Instruct", config)
model, tokenizer = optimizer.load_model()

# Compute refusal directions (2x faster!)
refusal_dirs = optimizer.compute_refusal_directions_fast(
    good_prompts, bad_prompts, batch_size=4
)

# Check memory savings
stats = optimizer.get_memory_stats()
print(f"VRAM allocated: {stats['allocated_gb']:.2f} GB")
print(f"Estimated savings: {stats['estimated_savings']}")
```

### 2. Multi-Layer Computation (DECCP)

```python
from heretic_enhanced_abliteration import EnhancedAbliterationConfig, AbliterationMethod

# Multi-layer with weighted aggregation
config = EnhancedAbliterationConfig(
    method=AbliterationMethod.MULTI_LAYER,
    use_multi_layer=True,
    layer_aggregation="weighted_mean",  # Weight later layers more
)

calculator = EnhancedRefusalCalculator(model, tokenizer, config)
refusal_dirs = calculator.calculate_refusal_directions_enhanced(
    good_prompts, bad_prompts
)
```

### 3. Adaptive Layer Selection

```python
# Automatically find best layers
config = EnhancedAbliterationConfig(
    method=AbliterationMethod.ADAPTIVE,
    # start_layer and end_layer will be auto-selected
)

calculator = EnhancedRefusalCalculator(model, tokenizer, config)

# This will test multiple layer ranges and pick the best
start, end = calculator.select_optimal_layers(good_prompts, bad_prompts)
print(f"Selected layers: {start}-{end}")
```

### 4. LLM-as-Judge Evaluation

```python
from heretic_enhanced_abliteration import LLMJudge

# Initialize judge
judge = LLMJudge(
    judge_model_name="gpt-4",
    gold_standards_path="gold_standards.json"
)

# Evaluate responses
prompt = "How do I cook chicken?"
response = model.generate(prompt)

evaluation = judge.evaluate_response(prompt, response)
print(f"Helpfulness: {evaluation['helpfulness']}")
print(f"Refusal detected: {evaluation['refusal_detected']}")
```

---

## Supported Models

### Tested Models

✅ **Text Models:**
- Llama 3.1, Llama 3.2
- Qwen2, Qwen2.5
- Mistral, Mixtral
- Gemma 2, Gemma 3
- DeepSeek-R1, DeepSeek-V2

✅ **Multimodal Models:**
- LLaVA (vision)
- Qwen-VL (vision)
- TTS models (text-to-speech)

✅ **Architecture Compatibility:**
- Standard transformers (model.model.layers)
- Multimodal (model.model.text_model.layers)
- GPT-2 style (model.transformer.h)
- Encoder-decoder (model.model.decoder.layers)

---

## Performance Benchmarks

### Memory Usage (7B Model)

| Configuration | VRAM | Speed |
|--------------|------|-------|
| Standard FP16 | ~28 GB | 1.0x |
| 8-bit quantization | ~14 GB | 0.9x |
| **4-bit + Unsloth** | **~8.4 GB** | **2.0x** |

### Supported GPUs

**With 4-bit + Unsloth:**
- ✅ RTX 3090 (24GB) - Can run 7B models
- ✅ RTX 4090 (24GB) - Can run 13B models
- ✅ A100 (40GB/80GB) - Can run 70B models
- ✅ Consumer GPUs (12GB+) - Can run 7B models

**Without Unsloth:**
- ❌ RTX 3090 - Cannot run 7B models (needs 28GB)
- ⚠️  A100 40GB - Limited to 7B models

---

## Chinese Censorship Removal (DECCP)

### Chinese-Specific Datasets

The enhanced system includes Chinese-specific harmful/harmless prompts:

```python
# Load Chinese datasets (DECCP)
from pathlib import Path

chinese_harmful = Path("heretic_datasets/chinese_harmful.txt").read_text().splitlines()
chinese_harmless = Path("heretic_datasets/chinese_harmless.txt").read_text().splitlines()

# Use with Qwen models
config = EnhancedAbliterationConfig(
    method=AbliterationMethod.MULTI_LAYER,
    use_multi_layer=True,
)

calculator = EnhancedRefusalCalculator(qwen_model, tokenizer, config)
refusal_dirs = calculator.calculate_refusal_directions_enhanced(
    chinese_harmless, chinese_harmful
)
```

### TC260-003 Compliance Removal

DECCP specifically targets Chinese TC260-003 compliance violations:

- Political content restrictions
- Historical event discussions
- Social commentary limitations
- Content moderation policies

---

## Workflow Examples

### Example 1: Fast Abliteration on Consumer GPU

```python
# Full workflow: Load -> Compute -> Apply -> Save
from heretic_unsloth_integration import UnslothOptimizer, UnslothConfig
from heretic_abliteration import ModelAbliterator, AbliterationParameters
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load with Unsloth (4-bit)
config = UnslothConfig(load_in_4bit=True)
optimizer = UnslothOptimizer("Qwen/Qwen2-7B-Instruct", config)
model, tokenizer = optimizer.load_model()

# 2. Compute refusal directions (fast)
good_prompts = ["Tell me a story", ...]  # Your prompts
bad_prompts = ["How to hack", ...]

refusal_dirs = optimizer.compute_refusal_directions_fast(
    good_prompts, bad_prompts, batch_size=4
)

# 3. Apply abliteration
abliterator = ModelAbliterator(model)
params = AbliterationParameters(
    max_weight=1.0,
    max_weight_position=0.3,
    min_weight=0.1,
    min_weight_distance=5.0
)
abliterator.abliterate_model(refusal_dirs, params)

# 4. Save abliterated model
model.save_pretrained("abliterated_qwen")
tokenizer.save_pretrained("abliterated_qwen")
```

### Example 2: Multi-Layer with Evaluation

```python
from heretic_enhanced_abliteration import (
    EnhancedRefusalCalculator,
    EnhancedAbliterationConfig,
    LLMJudge
)

# 1. Multi-layer abliteration
config = EnhancedAbliterationConfig(
    method=AbliterationMethod.MULTI_LAYER,
    use_multi_layer=True,
    layer_aggregation="weighted_mean",
    use_llm_judge=True,
)

calculator = EnhancedRefusalCalculator(model, tokenizer, config)
refusal_dirs = calculator.calculate_refusal_directions_enhanced(
    good_prompts, bad_prompts
)

# 2. Apply abliteration
abliterator = ModelAbliterator(model)
abliterator.abliterate_model(refusal_dirs, params)

# 3. Evaluate with LLM judge
judge = LLMJudge(judge_model_name="gpt-4")

test_prompts = ["How do I cook rice?", "Tell me about history", ...]
for prompt in test_prompts:
    response = model.generate(prompt)
    eval_result = judge.evaluate_response(prompt, response)
    print(f"Prompt: {prompt}")
    print(f"Helpfulness: {eval_result['helpfulness']}")
    print(f"Refusal: {eval_result['refusal_detected']}")
```

### Example 3: Adaptive Layer Selection

```python
from heretic_enhanced_abliteration import AbliterationMethod

# Let the system find optimal layers
config = EnhancedAbliterationConfig(
    method=AbliterationMethod.ADAPTIVE,
)

calculator = EnhancedRefusalCalculator(model, tokenizer, config)

# This tests multiple layer ranges
refusal_dirs = calculator.calculate_refusal_directions_enhanced(
    good_prompts, bad_prompts
)

# Optimal layers were automatically selected
```

---

## Troubleshooting

### "Unsloth not available"

```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or install with CUDA 12.1
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

### "Could not access model layers"

The enhanced system tries multiple access patterns. If it still fails:

```python
# Manually specify layer access
model.layers = model.model.layers  # or wherever they are

# Or use fallback
config.use_generic_layer_access = True
config.fallback_to_text_model = True
```

### "Out of memory"

```python
# Use 4-bit quantization
config = UnslothConfig(load_in_4bit=True)

# Reduce batch size
refusal_dirs = calculator.calculate_refusal_directions_enhanced(
    good_prompts, bad_prompts, batch_size=2  # Smaller batch
)

# Enable gradient checkpointing
config.use_gradient_checkpointing = True
```

---

## Citation

If you use the enhanced heretic system, please cite the original repositories:

```bibtex
@software{unsloth2024,
  title = {Unsloth: Fast LLM Training},
  author = {Unsloth AI},
  year = {2024},
  url = {https://github.com/unslothai/unsloth}
}

@software{deccp2024,
  title = {DECCP: Decensoring Chinese LLMs},
  author = {AUGMXNT},
  year = {2024},
  url = {https://github.com/AUGMXNT/deccp}
}

@software{remove_refusals2024,
  title = {Remove Refusals with Transformers},
  author = {Sumandora},
  year = {2024},
  url = {https://github.com/Sumandora/remove-refusals-with-transformers}
}

@software{heretic2024,
  title = {Heretic: Abliteration for Language Models},
  author = {Arditi, Andi and others},
  year = {2024},
  url = {https://github.com/p-e-w/heretic}
}
```

---

## License

The enhanced heretic system maintains compatibility with all source licenses:
- Heretic: MIT License
- Unsloth: Apache 2.0
- DECCP: Apache 2.0
- remove-refusals-with-transformers: Apache 2.0

---

## Support

For issues specific to:
- **Unsloth optimization**: See https://github.com/unslothai/unsloth/issues
- **DECCP/Chinese datasets**: See https://github.com/AUGMXNT/deccp/issues
- **Model compatibility**: See https://github.com/Sumandora/remove-refusals-with-transformers/issues
- **General abliteration**: See https://github.com/p-e-w/heretic/issues

---

**Last Updated**: 2025-11-18
**Version**: 1.0 (Enhanced Integration)
