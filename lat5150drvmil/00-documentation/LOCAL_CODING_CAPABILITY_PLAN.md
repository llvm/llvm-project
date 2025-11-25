# Local Coding Capability Plan

**Goal:** Enable local AI framework to write code without cloud dependency

**Date:** 2025-10-29
**Status:** Planning phase

---

## Current State

**Local AI (DeepSeek R1 1.5B):**
- ✅ Can explain code
- ✅ Can answer technical questions
- ⚠️ Limited code generation (small model)
- ⚠️ Not specialized for coding

**Cloud Options:**
- Gemini Pro: General purpose, not code-specialized
- OpenAI: Quota exceeded
- Claude Code: You're using it now (not local)

---

## Strategy: Local Code Generation

### Option 1: Larger Local Model (Recommended)

**Download specialized coding models:**

```bash
# CodeLlama 13B (good balance)
ollama pull codellama:13b-instruct

# DeepSeek Coder 6.7B (excellent for code)
ollama pull deepseek-coder:6.7b-instruct

# Phind CodeLlama 34B (best quality, slower)
ollama pull phind-codellama:34b

# Qwen Coder 14B (new, very good)
ollama pull qwen2.5-coder:14b
```

**Expected performance:**
- 13-14B models: 10-20 tok/sec, good quality code
- 34B model: 5-10 tok/sec, excellent quality
- All run locally, no guardrails, DSMIL-attested

### Option 2: Multi-Model Strategy

**Routing:**
- Code explanation → DeepSeek R1 1.5B (fast, adequate)
- Simple code snippets → DeepSeek Coder 6.7B (specialized)
- Complex code → CodeLlama 34B or Qwen 14B (high quality)
- Code review → CodeLlama 70B (already have it)

**Implementation:**
```python
# Update unified_orchestrator.py routing
def route_query(self, prompt, **kwargs):
    if is_code_task(prompt):
        if is_simple_snippet(prompt):
            return "local_coder_small"  # DeepSeek Coder 6.7B
        elif is_complex_code(prompt):
            return "local_coder_large"  # Qwen 14B or CodeLlama 34B
        else:
            return "local_coder_medium"  # CodeLlama 13B
    else:
        return "local_general"  # DeepSeek R1 1.5B
```

### Option 3: NPU-Accelerated Coding

**With Covert Edition NPU (49.4 TOPS):**
- Offload model inference to NPU
- CPU/GPU available for other tasks
- Potentially faster code generation

**Requires:**
- OpenVINO model conversion
- NPU-optimized model format
- Integration with Ollama or custom inference

---

## Recommended Implementation

### Phase 1: Download Specialized Models (30 min)

```bash
# Best balanced option
ollama pull deepseek-coder:6.7b-instruct  # 3.8GB, specialized for code
ollama pull qwen2.5-coder:14b             # 9GB, excellent quality

# Update dsmil_ai_engine.py
self.models = {
    "fast": "deepseek-r1:1.5b",           # General queries
    "code": "deepseek-coder:6.7b-instruct", # Code tasks
    "large": "qwen2.5-coder:14b",         # Complex code
    "review": "codellama:70b"             # Code review
}
```

### Phase 2: Code Task Detection (15 min)

```python
def is_code_task(query):
    """Detect if query is code-related"""
    code_keywords = [
        'write code', 'implement', 'function', 'class',
        'refactor', 'debug', 'fix bug', 'optimize',
        'create script', 'build', 'develop'
    ]
    return any(keyword in query.lower() for keyword in code_keywords)
```

### Phase 3: Test Code Generation (10 min)

```bash
# Test simple code task
python3 unified_orchestrator.py query "Write a Python function to check if a number is prime"

# Test complex code task
python3 unified_orchestrator.py query "Implement a Redis-like key-value store in Python with TTL support"

# Verify quality and speed
```

---

## Expected Results

### With DeepSeek Coder 6.7B + Qwen 14B

**Simple code tasks:**
- Model: DeepSeek Coder 6.7B
- Speed: 15-25 tok/sec
- Quality: Excellent for snippets, functions
- Time: 10-20s for typical function

**Complex code tasks:**
- Model: Qwen 2.5 Coder 14B
- Speed: 10-15 tok/sec
- Quality: Very high, good architecture
- Time: 30-60s for complex implementations

**Code review:**
- Model: CodeLlama 70B (already have)
- Speed: ~15 tok/sec
- Quality: Excellent analysis
- Time: 60-120s for thorough review

### With NPU Unlock (49.4 TOPS)

**Potential improvements:**
- 30-40% faster inference if NPU-accelerated
- More responsive for large models
- Better multi-tasking (NPU handles AI, CPU/GPU free)

---

## Integration with Current System

### Update unified_orchestrator.py

```python
class UnifiedAIOrchestrator:
    def __init__(self):
        self.local = DSMILAIEngine()

        # Specialized code models
        self.code_models = {
            "fast_code": "deepseek-coder:6.7b-instruct",
            "quality_code": "qwen2.5-coder:14b",
            "review": "codellama:70b"
        }

        self.gemini = GeminiAgent()
        self.openai = OpenAIAgent()

    def query(self, prompt, **kwargs):
        # Detect code task
        if self.is_code_task(prompt):
            # Route to specialized code model
            if self.is_complex_code(prompt):
                return self.local.generate(prompt, model="quality_code")
            else:
                return self.local.generate(prompt, model="fast_code")

        # Multimodal → Gemini
        elif kwargs.get('images') or kwargs.get('video'):
            return self.gemini.query(prompt, **kwargs)

        # Default → local general
        else:
            return self.local.generate(prompt, model_selection="fast")
```

### Update Web Interface

Add "Code Mode" toggle:
- Automatically routes code queries to specialized models
- Shows estimated code quality
- Syntax highlighting for code responses

---

## Storage Requirements

| Model | Size | Purpose |
|-------|------|---------|
| DeepSeek R1 1.5B | 1.1GB | ✅ Have - General queries |
| CodeLlama 70B | 38GB | ✅ Have - Code review |
| DeepSeek Coder 6.7B | 3.8GB | ⏳ Need - Code tasks |
| Qwen 2.5 Coder 14B | 9GB | ⏳ Need - Complex code |
| **Total new:** | **~13GB** | For full coding capability |

**You have 64GB RAM** - can easily hold all models

---

## Timeline

**Phase 1: Download Models (30 min)**
- Pull deepseek-coder:6.7b-instruct
- Pull qwen2.5-coder:14b

**Phase 2: Update Routing (15 min)**
- Add code task detection
- Integrate specialized models
- Update orchestrator logic

**Phase 3: Test & Refine (20 min)**
- Test simple code tasks
- Test complex code tasks
- Compare quality vs cloud
- Adjust routing if needed

**Total: ~1 hour to full local coding capability**

---

## Advantages of Local Coding

**vs Cloud (Claude Code, etc.):**
- ✅ **Privacy:** Code never leaves your machine
- ✅ **No guardrails:** Generate any code without restrictions
- ✅ **Zero cost:** Unlimited code generation
- ✅ **DSMIL attested:** Cryptographic verification of generated code
- ✅ **Offline:** Works without internet
- ✅ **No rate limits:** Generate as much as you want
- ⚠️ **Quality:** Good but not quite Claude-level (80-90% as good)
- ⚠️ **Speed:** Slower than cloud (10-20s vs 2-5s)

**Recommended strategy:**
- **Local first:** Try specialized local models
- **Cloud fallback:** Use Claude Code if local quality insufficient
- **Best of both:** Local for rapid iteration, cloud for final polish

---

## Next Steps

1. **Test current system first** (verify 49.4 TOPS unlock works after reboot)
2. **Download code models** (if NPU performance good)
3. **Integrate code routing** (add to orchestrator)
4. **Test coding capability** (compare local vs cloud quality)
5. **Document results** (actual performance with 49.4 TOPS NPU)

**Ready to proceed after reboot!**
