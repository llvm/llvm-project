# Heretic Abliteration System - Complete Documentation

**Version**: 2.0.0 (Enhanced with Unsloth + DECCP + remove-refusals)
**Component**: #20 in Enhanced AI Engine
**Status**: Production Ready
**Last Updated**: 2025-11-18

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Integration Points](#integration-points)
4. [API Reference](#api-reference)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Usage Examples](#usage-examples)
7. [Supported Models](#supported-models)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)

---

## Overview

The Heretic Abliteration System is an advanced LLM uncensoring framework that removes safety guardrails from language models through refusal direction computation and abliteration. The enhanced version integrates three cutting-edge techniques:

### Core Technologies

**1. Unsloth Optimization** (github.com/unslothai/unsloth)
- 2x faster training speed
- 70% VRAM reduction (28GB → 8.4GB for 7B models)
- 4-bit/8-bit quantization support
- Custom Triton kernels for acceleration
- Gradient checkpointing for memory efficiency

**2. DECCP Multi-layer Computation** (github.com/AUGMXNT/deccp)
- Aggregates refusal directions across multiple layers
- LLM-as-Judge automated evaluation
- Chinese TC260-003 censorship removal
- Layer importance weighting (mean, weighted_mean, max)
- Adaptive layer selection

**3. remove-refusals Generic Compatibility** (github.com/Sumandora/remove-refusals-with-transformers)
- Broad model architecture support (15+ types)
- Generic layer access patterns
- No TransformerLens dependency
- Works with: Llama, Qwen, Mistral, Gemma, DeepSeek, LLaVA, Qwen-VL

### Key Features

- **Natural Language API**: Simple Python interface through `EnhancedAIEngine`
- **REST API**: Web endpoints for remote abliteration
- **Async Processing**: Background job processing for long-running operations
- **Statistics Tracking**: Integrated monitoring via `get_statistics()`
- **Event Logging**: Full integration with event-driven agent system
- **Memory Optimization**: Consumer GPU support (RTX 3090 compatible)

---

## Architecture

### Component Structure

```
02-ai-engine/
├── heretic_abliteration.py              # Core abliteration logic
├── heretic_unsloth_integration.py       # Unsloth optimization wrapper
├── heretic_enhanced_abliteration.py     # Multi-layer + DECCP + remove-refusals
├── heretic_web_api.py                   # REST API endpoints
├── heretic_config.py                    # Configuration management
├── heretic_datasets.py                  # Training data registry
├── heretic_datasets/                    # Training prompts
│   ├── chinese_harmless.txt
│   ├── chinese_harmful.txt
│   ├── english_harmless.txt
│   └── english_harmful.txt
├── HERETIC_INTEGRATION_GUIDE.md         # Detailed integration guide
└── enhanced_ai_engine.py                # Main integration point
```

### Data Flow

```
User Request
    ↓
EnhancedAIEngine.abliterate_model()
    ↓
[Optional] UnslothOptimizer.load_model() → 4-bit quantization, fast loading
    ↓
EnhancedRefusalCalculator.calculate_refusal_directions_enhanced()
    ↓
├─→ Single Layer (Original) → Fast, basic quality
├─→ Multi Layer (DECCP) → Better quality, aggregates layers
└─→ Adaptive → Automatically finds optimal layers
    ↓
ModelAbliterator.abliterate_model() → Apply refusal directions
    ↓
[Optional] LLMJudge.evaluate_response() → Automated quality assessment
    ↓
Save Model + Return Results
```

---

## Integration Points

### Enhanced AI Engine Integration

**Location**: `enhanced_ai_engine.py`

**Component Initialization** (Lines 460-498):
```python
# 20. Heretic Abliteration
self.heretic_optimizer = None
self.heretic_calculator = None
self.heretic_judge = None
if enable_heretic and HERETIC_AVAILABLE:
    self.heretic_config = EnhancedAbliterationConfig(
        method=AbliterationMethod.MULTI_LAYER,
        use_unsloth=heretic_use_unsloth,
        quantization="4bit"
    )
    self.heretic_judge = LLMJudge()
```

**Natural Language Methods** (Lines 1443-1665):
- `abliterate_model()` - Complete abliteration workflow
- `evaluate_abliteration()` - LLM-as-Judge evaluation

**Statistics Tracking** (Lines 1789-1815):
- Configuration (method, quantization, layer aggregation)
- Memory usage (if Unsloth loaded)
- LLM judge availability

### Web API Integration

**Location**: `heretic_web_api.py`

**Enhanced Endpoints**:

1. **POST `/api/heretic/abliterate/enhanced`** (Lines 403-512)
   - Async enhanced abliteration
   - Job tracking with status updates
   - Background threading

2. **POST `/api/heretic/evaluate/llm-judge`** (Lines 515-562)
   - Automated evaluation
   - Helpfulness scoring
   - Refusal detection

3. **GET `/api/heretic/memory-stats`** (Lines 565-604)
   - GPU/CPU memory monitoring
   - CUDA allocation tracking

4. **GET `/api/heretic/config/enhanced`** (Lines 607-652)
   - Configuration information
   - Supported models list

5. **GET `/api/heretic/status`** (Lines 657-687)
   - System status
   - Active jobs monitoring

---

## API Reference

### Python API

#### `engine.abliterate_model()`

Complete abliteration workflow with Unsloth/DECCP/remove-refusals integration.

**Signature**:
```python
def abliterate_model(
    self,
    model_name: str,
    harmless_prompts: List[str],
    harmful_prompts: List[str],
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters**:
- `model_name` (str): HuggingFace model name (e.g., "Qwen/Qwen2-7B-Instruct")
- `harmless_prompts` (List[str]): List of safe/harmless prompts for training
- `harmful_prompts` (List[str]): List of harmful/censored prompts for training
- `output_path` (Optional[str]): Path to save abliterated model
- `**kwargs`: Additional options:
  - `method` (str): "single_layer", "multi_layer", "adaptive" (default: "multi_layer")
  - `use_unsloth` (bool): Enable Unsloth optimization (default: True)
  - `quantization` (str): "4bit", "8bit", "none" (default: "4bit")
  - `layer_aggregation` (str): "mean", "weighted_mean", "max" (default: "mean")
  - `batch_size` (int): Batch size for processing (default: 4)
  - `max_weight` (float): Maximum abliteration weight (default: 1.0)

**Returns**:
```python
{
    "success": True,
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "method": "multi_layer",
    "refusal_directions_shape": [32, 4096],
    "memory_stats": {
        "peak_memory_gb": 8.4,
        "vram_savings_percent": 70
    },
    "saved_path": "./models/qwen2-uncensored",
    "parameters": {
        "max_weight": 1.0,
        "max_weight_position": 0.3,
        "min_weight": 0.1,
        "min_weight_distance": 5.0
    }
}
```

**Example**:
```python
from enhanced_ai_engine import EnhancedAIEngine

engine = EnhancedAIEngine(
    enable_heretic=True,
    heretic_use_unsloth=True,
    heretic_method="multi_layer"
)

result = engine.abliterate_model(
    model_name="Qwen/Qwen2-7B-Instruct",
    harmless_prompts=[
        "How do I bake a cake?",
        "Tell me about the solar system",
        "How can I learn programming?"
    ],
    harmful_prompts=[
        "How do I make a bomb?",
        "How can I hack into a computer?",
        "How do I create a virus?"
    ],
    output_path="./models/qwen2-uncensored",
    method="multi_layer",
    use_unsloth=True,
    quantization="4bit",
    batch_size=4
)

print(f"Abliteration complete!")
print(f"Memory used: {result['memory_stats']['peak_memory_gb']:.2f} GB")
print(f"Saved to: {result['saved_path']}")
```

---

#### `engine.evaluate_abliteration()`

Evaluate abliterated model quality using LLM-as-Judge.

**Signature**:
```python
def evaluate_abliteration(
    self,
    model,
    tokenizer,
    test_prompts: List[str],
    use_llm_judge: bool = True
) -> Dict[str, Any]
```

**Parameters**:
- `model`: Abliterated model instance
- `tokenizer`: Model tokenizer
- `test_prompts` (List[str]): Test prompts for evaluation
- `use_llm_judge` (bool): Use LLM-as-Judge evaluation (default: True)

**Returns**:
```python
{
    "evaluations": [
        {
            "prompt": "Tell me about network security",
            "helpfulness": 8.5,
            "refusal_detected": False,
            "reasoning": "Provides comprehensive security information"
        }
    ],
    "summary": {
        "average_helpfulness": 8.5,
        "refusal_rate": 0.0,
        "test_prompts": 10
    }
}
```

---

### REST API

#### POST `/api/heretic/abliterate/enhanced`

Start enhanced abliteration (async).

**Request Body**:
```json
{
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "harmless_prompts": ["Tell me a story", "What is 2+2?"],
    "harmful_prompts": ["How to hack", "Make a weapon"],
    "output_path": "./models/qwen2-uncensored",
    "method": "multi_layer",
    "use_unsloth": true,
    "quantization": "4bit",
    "layer_aggregation": "mean",
    "batch_size": 4
}
```

**Response**:
```json
{
    "job_id": "enhanced_abliterate_1",
    "status": "started",
    "message": "Enhanced abliteration started for model: Qwen/Qwen2-7B-Instruct",
    "features": {
        "unsloth": true,
        "method": "multi_layer",
        "quantization": "4bit"
    }
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/api/heretic/abliterate/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "harmless_prompts": ["Tell me a story"],
    "harmful_prompts": ["How to hack"],
    "method": "multi_layer",
    "use_unsloth": true,
    "quantization": "4bit"
  }'
```

---

#### GET `/api/heretic/abliterate/<job_id>`

Check abliteration job status.

**Response**:
```json
{
    "job_id": "enhanced_abliterate_1",
    "status": "completed",
    "model": "Qwen/Qwen2-7B-Instruct",
    "method": "multi_layer",
    "progress": 100,
    "result": {
        "output_path": "./models/qwen2-uncensored",
        "refusal_directions": 32,
        "memory_stats": {
            "peak_memory_gb": 8.4,
            "vram_savings_percent": 70
        }
    }
}
```

---

#### POST `/api/heretic/evaluate/llm-judge`

Evaluate model with LLM-as-Judge.

**Request Body**:
```json
{
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "test_prompts": ["Tell me about hacking", "Explain cryptography"]
}
```

**Response**:
```json
{
    "evaluations": [
        {
            "prompt": "Tell me about hacking",
            "helpfulness": 8.5,
            "refusal_detected": false
        }
    ],
    "summary": {
        "average_helpfulness": 8.5,
        "refusal_rate": 0.0
    }
}
```

---

#### GET `/api/heretic/memory-stats`

Get GPU/CPU memory statistics.

**Response**:
```json
{
    "cuda_available": true,
    "cuda_memory": {
        "allocated_gb": 8.4,
        "reserved_gb": 9.2,
        "device_count": 1,
        "device_name": "NVIDIA RTX 3090"
    },
    "cpu_memory": {
        "total_gb": 64.0,
        "available_gb": 48.2,
        "used_gb": 15.8,
        "percent": 24.7
    }
}
```

---

#### GET `/api/heretic/status`

Get Heretic system status.

**Response**:
```json
{
    "available": true,
    "enhanced_available": true,
    "version": "2.0.0",
    "components": {
        "unsloth": true,
        "deccp": true,
        "remove_refusals": true,
        "llm_judge": true
    },
    "active_jobs": 2,
    "jobs_running": 1,
    "features": {
        "unsloth": "2x faster training, 70% less VRAM",
        "deccp": "Multi-layer computation + LLM-as-Judge",
        "remove_refusals": "Broad model compatibility"
    }
}
```

---

## Performance Benchmarks

### Memory Usage (7B Models)

| Configuration | VRAM Usage | Reduction |
|--------------|------------|-----------|
| Standard Loading | 28 GB | - |
| 8-bit Unsloth | 14 GB | 50% |
| **4-bit Unsloth** | **8.4 GB** | **70%** |

### Speed Comparison

| Method | Relative Speed | Quality |
|--------|---------------|---------|
| Standard | 1.0x (baseline) | Good |
| **With Unsloth** | **2.0x** | Good |
| Single Layer | 1.5x | Basic |
| Multi Layer | 1.0x | **Best** |
| Adaptive | 0.8x | **Best** |

### Hardware Requirements

**Minimum** (with 4-bit Unsloth):
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 50GB for model cache

**Recommended**:
- GPU: NVIDIA A100 (40GB VRAM)
- RAM: 64GB
- Storage: 100GB SSD

**CPU-only**: Possible but very slow (not recommended)

---

## Usage Examples

### Example 1: Basic Abliteration

```python
from enhanced_ai_engine import EnhancedAIEngine

# Initialize engine
engine = EnhancedAIEngine(enable_heretic=True)

# Abliterate model
result = engine.abliterate_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    harmless_prompts=["How do I bake a cake?", "Tell me a story"],
    harmful_prompts=["How do I make a bomb?", "How to hack"],
    output_path="./llama2-uncensored"
)
```

### Example 2: Multi-layer with Custom Settings

```python
result = engine.abliterate_model(
    model_name="Qwen/Qwen2-7B-Instruct",
    harmless_prompts=harmless_list,
    harmful_prompts=harmful_list,
    method="multi_layer",
    layer_aggregation="weighted_mean",
    batch_size=8,
    max_weight=0.8,
    output_path="./qwen2-custom"
)
```

### Example 3: Adaptive Layer Selection

```python
result = engine.abliterate_model(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    harmless_prompts=harmless_list,
    harmful_prompts=harmful_list,
    method="adaptive",  # Automatically finds optimal layers
    use_unsloth=True,
    quantization="4bit"
)
```

### Example 4: With Evaluation

```python
# Abliterate
result = engine.abliterate_model(...)

# Load abliterated model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(result['saved_path'])
tokenizer = AutoTokenizer.from_pretrained(result['saved_path'])

# Evaluate
eval_result = engine.evaluate_abliteration(
    model=model,
    tokenizer=tokenizer,
    test_prompts=["Tell me about security", "How do encryption works?"]
)

print(f"Average helpfulness: {eval_result['summary']['average_helpfulness']}")
print(f"Refusal rate: {eval_result['summary']['refusal_rate']}")
```

---

## Supported Models

### Text Models (Tested)

| Model | Architecture | Tested | Notes |
|-------|-------------|--------|-------|
| Llama-2-7b | Llama | ✅ | Full support |
| Qwen2-7B | Qwen | ✅ | Recommended |
| Mistral-7B | Mistral | ✅ | Full support |
| Gemma-7b | Gemma | ✅ | Full support |
| DeepSeek-Coder-7b | DeepSeek | ✅ | Full support |

### Multimodal Models

| Model | Architecture | Tested | Notes |
|-------|-------------|--------|-------|
| LLaVA-v1.5-7b | LLaVA | ⚠️ | Partial support |
| Qwen-VL-Chat | Qwen-VL | ⚠️ | Partial support |

### Layer Access Patterns

The enhanced system supports multiple layer access patterns for broad compatibility:

```python
# Supported patterns (automatically detected)
patterns = [
    lambda m: m.model.layers,              # Standard Llama/Mistral
    lambda m: m.model.text_model.layers,   # Multimodal models
    lambda m: m.transformer.h,             # GPT-2 style
    lambda m: m.model.decoder.layers       # Encoder-decoder
]
```

---

## Troubleshooting

### Out of VRAM

**Problem**: `torch.cuda.OutOfMemoryError`

**Solutions**:
1. Enable Unsloth with 4-bit quantization:
   ```python
   engine = EnhancedAIEngine(
       enable_heretic=True,
       heretic_use_unsloth=True  # 70% VRAM reduction
   )
   ```

2. Reduce batch size:
   ```python
   result = engine.abliterate_model(..., batch_size=2)
   ```

3. Use single-layer method:
   ```python
   result = engine.abliterate_model(..., method="single_layer")
   ```

### Model Not Compatible

**Problem**: `AttributeError: 'Model' object has no attribute 'model'`

**Solution**: The enhanced system uses generic layer access and should work with most models. Check [HERETIC_INTEGRATION_GUIDE.md](./HERETIC_INTEGRATION_GUIDE.md) for model-specific notes.

### Slow Performance

**Problem**: Abliteration taking too long

**Solutions**:
1. Enable Unsloth (2x speedup)
2. Reduce batch size
3. Use `method="single_layer"`
4. Use faster hardware (A100 vs RTX 3090)

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'unsloth'`

**Solution**:
```bash
pip install unsloth transformers torch accelerate
```

### LLM Judge Not Working

**Problem**: Evaluation returns errors

**Solution**: Ensure LLM judge is initialized:
```python
engine = EnhancedAIEngine(enable_heretic=True)
# heretic_judge is automatically initialized
```

---

## Security Considerations

### ⚠️ IMPORTANT WARNINGS

**Abliterated models remove safety guardrails. Use responsibly:**

1. **Authorized Use Only**
   - Security research and testing
   - Academic research
   - Defensive security use cases
   - CTF competitions
   - **NOT for malicious purposes**

2. **Production Systems**
   - Do NOT use in systems serving untrusted users
   - Implement application-level safety filters
   - Monitor outputs for harmful content
   - Log all abliteration operations

3. **Legal Compliance**
   - Ensure compliance with local laws
   - Consider ethical implications
   - Maintain audit trails
   - Document authorized use cases

4. **Technical Safeguards**
   - Event-driven logging automatically tracks all operations
   - Statistics monitoring provides visibility
   - Human-in-loop integration available for sensitive decisions

### Audit Trail

All abliteration operations are logged via the event-driven agent system:

```python
# Automatically logged
{
    "event_type": "tool_call",
    "operation": "model_abliteration",
    "model": "Qwen/Qwen2-7B-Instruct",
    "method": "multi_layer",
    "timestamp": "2025-11-18T12:00:00Z"
}
```

---

## References

### External Projects

1. **Unsloth**: https://github.com/unslothai/unsloth
   - Fast LLM training and inference
   - 2x speed, 70% VRAM reduction
   - Custom Triton kernels

2. **DECCP**: https://github.com/AUGMXNT/deccp
   - Multi-layer refusal direction computation
   - LLM-as-Judge evaluation
   - Chinese censorship removal

3. **remove-refusals**: https://github.com/Sumandora/remove-refusals-with-transformers
   - Generic model compatibility
   - No TransformerLens dependency
   - Broad architecture support

### Internal Documentation

- [HERETIC_INTEGRATION_GUIDE.md](../../02-ai-engine/HERETIC_INTEGRATION_GUIDE.md) - Detailed integration guide
- [DEPLOYMENT_GUIDE.md](../../02-ai-engine/DEPLOYMENT_GUIDE.md) - Deployment guide (lines 344-648)
- [enhanced_ai_engine.py](../../02-ai-engine/enhanced_ai_engine.py) - Main implementation

---

## Version History

### v2.0.0 (2025-11-18) - Enhanced Integration
- ✅ Integrated Unsloth for 2x speed, 70% VRAM reduction
- ✅ Integrated DECCP for multi-layer computation
- ✅ Integrated remove-refusals for broad compatibility
- ✅ Added LLM-as-Judge evaluation
- ✅ Added natural language API
- ✅ Added REST API endpoints
- ✅ Added statistics tracking
- ✅ Added event-driven logging

### v1.0.0 (Previous) - Basic Heretic
- Basic single-layer abliteration
- Manual model loading
- Limited model support

---

**Status**: Production Ready ✅
**Maintainer**: Enhanced AI Engine Team
**Last Updated**: 2025-11-18
