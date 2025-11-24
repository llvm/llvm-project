# DSMIL AI-Assisted Compilation
**Integration Guide for DSMIL Layers 3-9 AI Advisors**

Version: 1.0
Last Updated: 2025-11-24

---

## Overview

DSLLVM integrates with the DSMIL AI architecture (Layers 3-9, 48 AI devices, ~1338 TOPS INT8) to provide intelligent compilation assistance while maintaining deterministic, auditable builds.

**AI Integration Principles**:
1. **Advisory, not authoritative**: AI suggests; deterministic passes verify
2. **Auditable**: All AI interactions logged with timestamps and versions
3. **Fallback-safe**: Classical heuristics used if AI unavailable
4. **Mode-configurable**: `off`, `local`, `advisor`, `lab` modes

---

## 1. AI Advisor Architecture

### 1.1 Overview

```
┌─────────────────────────────────────────────────────┐
│                  DSLLVM Compiler                    │
│                                                     │
│  ┌─────────────┐      ┌─────────────┐             │
│  │  IR Module  │─────→│ AI Advisor  │             │
│  │  Summary    │      │   Passes    │             │
│  └─────────────┘      └──────┬──────┘             │
│                              │                      │
│                              ↓                      │
│                   *.dsmilai_request.json            │
└──────────────────────────┬──────────────────────────┘
                           │
                           ↓
    ┌──────────────────────────────────────────┐
    │         DSMIL AI Service Layer          │
    │                                          │
    │  ┌──────────┐  ┌───────────┐  ┌───────┐│
    │  │ Layer 7  │  │  Layer 8  │  │ L5/6  ││
    │  │ LLM      │  │ Security  │  │ Perf  ││
    │  │ Advisor  │  │    AI     │  │ Model ││
    │  └────┬─────┘  └─────┬─────┘  └───┬───┘│
    │       │              │              │   │
    │       └──────────────┴──────────────┘   │
    │                     │                    │
    │          *.dsmilai_response.json         │
    └─────────────────────┬────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────┐
│                  DSLLVM Compiler                    │
│                                                     │
│  ┌──────────────────┐      ┌──────────────────┐   │
│  │  AI Response     │─────→│  Deterministic   │   │
│  │  Parser          │      │  Verification    │   │
│  └──────────────────┘      └──────┬───────────┘   │
│                                    │               │
│                                    ↓               │
│                         Updated IR + Metadata      │
└─────────────────────────────────────────────────────┘
```

### 1.2 Integration Points

| Pass | Layer | Device | Purpose | Mode |
|------|-------|--------|---------|------|
| `dsmil-ai-advisor-annotate` | 7 | 47 | Code annotation suggestions | advisor, lab |
| `dsmil-ai-security-scan` | 8 | 80-87 | Security risk analysis | advisor, lab |
| `dsmil-ai-perf-forecast` | 5-6 | 50-59 | Performance prediction | advisor (tool) |
| `DsmilAICostModelPass` | N/A | local | ML cost models (ONNX) | local, advisor, lab |

---

## 2. Request/Response Protocol

### 2.1 Request Schema: `*.dsmilai_request.json`

```json
{
  "schema": "dsmilai-request-v1",
  "version": "1.0",
  "timestamp": "2025-11-24T15:30:45Z",
  "compiler": {
    "name": "dsmil-clang",
    "version": "19.0.0-dsmil",
    "target": "x86_64-dsmil-meteorlake-elf"
  },
  "build_config": {
    "mode": "advisor",
    "policy": "production",
    "ai_mode": "advisor",
    "optimization_level": "-O3"
  },
  "module": {
    "name": "llm_inference.c",
    "path": "/workspace/src/llm_inference.c",
    "hash_sha384": "d4f8c9a3e2b1f7c6...",
    "source_lines": 1247,
    "functions": 23,
    "globals": 8
  },
  "advisor_request": {
    "advisor_type": "l7_llm",  // or "l8_security", "l5_perf"
    "request_id": "uuid-1234-5678-...",
    "priority": "normal",  // "low", "normal", "high"
    "goals": {
      "latency_target_ms": 100,
      "power_budget_w": 120,
      "security_posture": "high",
      "accuracy_target": 0.95
    }
  },
  "ir_summary": {
    "functions": [
      {
        "name": "llm_decode_step",
        "mangled_name": "_Z15llm_decode_stepPKfPf",
        "loc": "llm_inference.c:127",
        "basic_blocks": 18,
        "instructions": 342,
        "calls": ["matmul_kernel", "softmax", "layer_norm"],
        "loops": 3,
        "max_loop_depth": 2,
        "memory_accesses": {
          "loads": 156,
          "stores": 48,
          "estimated_bytes": 1048576
        },
        "vectorization": {
          "auto_vectorized": true,
          "vector_width": 256,
          "vector_isa": "AVX2"
        },
        "existing_metadata": {
          "dsmil_layer": null,
          "dsmil_device": null,
          "dsmil_stage": null,
          "dsmil_clearance": null
        },
        "cfg_features": {
          "cyclomatic_complexity": 12,
          "branch_density": 0.08,
          "dominance_depth": 4
        }
      }
    ],
    "globals": [
      {
        "name": "attention_weights",
        "type": "const float[4096][4096]",
        "size_bytes": 67108864,
        "initializer": true,
        "constant": true,
        "existing_metadata": {
          "dsmil_hot_model": false,
          "dsmil_kv_cache": false
        }
      }
    ],
    "call_graph": {
      "nodes": 23,
      "edges": 47,
      "strongly_connected_components": 1,
      "max_call_depth": 5
    },
    "data_flow": {
      "untrusted_sources": ["user_input_buffer"],
      "sensitive_sinks": ["crypto_sign", "network_send"],
      "flows": [
        {
          "from": "user_input_buffer",
          "to": "process_input",
          "path_length": 3,
          "sanitized": false
        }
      ]
    }
  },
  "context": {
    "project_type": "llm_inference_server",
    "deployment_target": "layer7_production",
    "previous_builds": {
      "last_build_hash": "a1b2c3d4...",
      "performance_history": {
        "avg_latency_ms": 87.3,
        "p99_latency_ms": 142.1,
        "throughput_qps": 234
      }
    }
  }
}
```

### 2.2 Response Schema: `*.dsmilai_response.json`

```json
{
  "schema": "dsmilai-response-v1",
  "version": "1.0",
  "timestamp": "2025-11-24T15:30:47Z",
  "request_id": "uuid-1234-5678-...",
  "advisor": {
    "type": "l7_llm",
    "model": "Llama-3-7B-INT8",
    "version": "2024.11",
    "device": 47,
    "layer": 7,
    "confidence_threshold": 0.75
  },
  "processing": {
    "duration_ms": 1834,
    "tokens_processed": 4523,
    "inference_cost_tops": 12.4
  },
  "suggestions": {
    "annotations": [
      {
        "target": "function:llm_decode_step",
        "attributes": [
          {
            "name": "dsmil_layer",
            "value": 7,
            "confidence": 0.92,
            "rationale": "Function performs AI inference operations typical of Layer 7 (AI/ML). Calls matmul_kernel and layer_norm which are LLM primitives."
          },
          {
            "name": "dsmil_device",
            "value": 47,
            "confidence": 0.88,
            "rationale": "High memory bandwidth requirements (1 MB per call) and vectorized compute suggest NPU (Device 47) placement."
          },
          {
            "name": "dsmil_stage",
            "value": "quantized",
            "confidence": 0.95,
            "rationale": "Code uses INT8 data types and quantized attention weights, indicating quantized inference stage."
          },
          {
            "name": "dsmil_hot_model",
            "value": true,
            "confidence": 0.90,
            "rationale": "attention_weights accessed in hot loop; should be marked dsmil_hot_model for optimal placement."
          }
        ]
      }
    ],
    "refactoring": [
      {
        "target": "function:llm_decode_step",
        "suggestion": "split_function",
        "confidence": 0.78,
        "description": "Function has high cyclomatic complexity (12). Consider splitting into llm_decode_step_prepare and llm_decode_step_execute.",
        "impact": {
          "maintainability": "high",
          "performance": "neutral",
          "security": "neutral"
        }
      }
    ],
    "security_hints": [
      {
        "target": "data_flow:user_input_buffer→process_input",
        "severity": "medium",
        "confidence": 0.85,
        "finding": "Untrusted input flows into processing without sanitization",
        "recommendation": "Mark user_input_buffer with __attribute__((dsmil_untrusted_input)) and add validation in process_input",
        "cwe": "CWE-20: Improper Input Validation"
      }
    ],
    "performance_hints": [
      {
        "target": "function:matmul_kernel",
        "hint": "device_offload",
        "confidence": 0.87,
        "description": "Matrix multiplication with dimensions 4096x4096 is well-suited for NPU/GPU offload",
        "expected_speedup": 3.2,
        "power_impact": "+8W"
      }
    ],
    "pipeline_tuning": [
      {
        "pass": "vectorizer",
        "parameter": "vectorization_factor",
        "current_value": 8,
        "suggested_value": 16,
        "confidence": 0.81,
        "rationale": "AVX-512 available on Meteor Lake; widening vectorization factor from 8 to 16 can improve throughput by ~18%"
      }
    ]
  },
  "diagnostics": {
    "warnings": [
      "Function llm_decode_step has no dsmil_clearance attribute. Defaulting to 0x00000000 may cause layer transition issues."
    ],
    "info": [
      "Model attention_weights is 64 MB. Consider compression or tiling for memory efficiency."
    ]
  },
  "metadata": {
    "model_hash_sha384": "f7a3b9c2...",
    "inference_session_id": "session-9876-5432",
    "fallback_used": false,
    "cached_response": false
  }
}
```

---

## 3. Layer 7 LLM Advisor

### 3.1 Capabilities

**Device**: Layer 7, Device 47 (NPU primary)
**Model**: Llama-3-7B-INT8 (~7B parameters, INT8 quantized)
**Context**: Up to 8192 tokens

**Specialized For**:
- Code annotation inference
- DSMIL layer/device/stage suggestion
- Refactoring recommendations
- Explainability (generate human-readable rationales)

### 3.2 Prompt Template

```
You are an expert compiler assistant for the DSMIL architecture. Analyze the following LLVM IR summary and suggest appropriate DSMIL attributes.

DSMIL Architecture:
- 9 layers (3-9): Hardware → Kernel → Drivers → Crypto → Network → System → Middleware → Application → UI
- 104 devices (0-103): Including 48 AI devices across layers 3-9
- Device 47: Primary NPU for AI/ML workloads

Function to analyze:
Name: llm_decode_step
Location: llm_inference.c:127
Basic blocks: 18
Instructions: 342
Calls: matmul_kernel, softmax, layer_norm
Memory accesses: 156 loads, 48 stores, ~1 MB
Vectorization: AVX2 (256-bit)

Project context:
- Type: LLM inference server
- Deployment: Layer 7 production
- Performance target: <100ms latency

Suggest:
1. dsmil_layer (3-9)
2. dsmil_device (0-103)
3. dsmil_stage (pretrain/finetune/quantized/serve/etc.)
4. Other relevant attributes (dsmil_hot_model, dsmil_kv_cache, etc.)

Provide rationale for each suggestion with confidence scores (0.0-1.0).
```

### 3.3 Integration Flow

```
1. DSLLVM Pass: dsmil-ai-advisor-annotate
   ↓
2. Generate IR summary from module
   ↓
3. Serialize to *.dsmilai_request.json
   ↓
4. Submit to Layer 7 LLM service (HTTP/gRPC/Unix socket)
   ↓
5. L7 service processes with Llama-3-7B-INT8
   ↓
6. Returns *.dsmilai_response.json
   ↓
7. Parse response in DSLLVM
   ↓
8. For each suggestion:
   a. Check confidence >= threshold (default 0.75)
   b. Validate against DSMIL constraints (layer bounds, device ranges)
   c. If valid: add to IR metadata with !dsmil.suggested.* namespace
   d. If invalid: log warning
   ↓
9. Downstream passes (dsmil-layer-check, etc.) validate suggestions
   ↓
10. Only suggestions passing verification are applied to final binary
```

---

## 4. Layer 8 Security AI Advisor

### 4.1 Capabilities

**Device**: Layer 8, Devices 80-87 (~188 TOPS combined)
**Models**: Ensemble of security-focused ML models
- Taint analysis model (transformer-based)
- Vulnerability pattern detector (CNN)
- Side-channel risk estimator (RNN)

**Specialized For**:
- Untrusted input flow analysis
- Vulnerability pattern detection (buffer overflows, use-after-free, etc.)
- Side-channel risk assessment
- Sandbox profile recommendations

### 4.2 Request Extensions

Additional fields for L8 security advisor:

```json
{
  "advisor_request": {
    "advisor_type": "l8_security"
  },
  "security_context": {
    "threat_model": "internet_facing",
    "attack_surface": ["network", "ipc", "file_io"],
    "sensitivity_level": "high",
    "compliance": ["CNSA2.0", "FIPS140-3"]
  },
  "taint_sources": [
    {
      "name": "user_input_buffer",
      "type": "network_socket",
      "trusted": false
    }
  ],
  "sensitive_sinks": [
    {
      "name": "crypto_sign",
      "type": "cryptographic_operation",
      "requires_validation": true
    }
  ]
}
```

### 4.3 Response Extensions

```json
{
  "suggestions": {
    "security_hints": [
      {
        "target": "function:process_input",
        "severity": "high",
        "confidence": 0.91,
        "finding": "Input validation bypass potential",
        "recommendation": "Add bounds checking before memcpy at line 234",
        "cwe": "CWE-120: Buffer Copy without Checking Size of Input",
        "cvss_score": 7.5,
        "exploit_complexity": "low"
      }
    ],
    "sandbox_recommendations": [
      {
        "target": "binary",
        "profile": "l7_llm_worker_strict",
        "rationale": "Function process_input handles untrusted network data. Recommend strict sandbox with no network egress after initialization.",
        "confidence": 0.88
      }
    ],
    "side_channel_risks": [
      {
        "target": "function:crypto_compare",
        "risk_type": "timing",
        "severity": "medium",
        "confidence": 0.79,
        "description": "String comparison may leak timing information",
        "mitigation": "Use constant-time comparison (e.g., crypto_memcmp)"
      }
    ]
  }
}
```

### 4.4 Integration Modes

**Mode 1: Offline (embedded model)**
```bash
# Use pre-trained model shipped with DSLLVM
dsmil-clang -fpass-pipeline=dsmil-default \
  --ai-mode=local \
  -mllvm -dsmil-security-model=/opt/dsmil/models/security_v1.onnx \
  -o output input.c
```

**Mode 2: Online (L8 service)**
```bash
# Query external L8 security service
export DSMIL_L8_SECURITY_URL=http://l8-security.dsmil.internal:8080
dsmil-clang -fpass-pipeline=dsmil-default \
  --ai-mode=advisor \
  -o output input.c
```

---

## 5. Layer 5/6 Performance Forecasting

### 5.1 Capabilities

**Devices**: Layer 5-6, Devices 50-59 (predictive analytics)
**Models**: Time-series forecasting + scenario simulation

**Specialized For**:
- Runtime performance prediction
- Hot path identification
- Resource utilization forecasting
- Power/latency tradeoff analysis

### 5.2 Tool: `dsmil-ai-perf-forecast`

```bash
# Offline tool (not compile-time pass)
dsmil-ai-perf-forecast \
  --binary llm_worker \
  --dsmilmap llm_worker.dsmilmap \
  --history-dir /var/dsmil/metrics/ \
  --scenario production_load \
  --output perf_forecast.json
```

### 5.3 Input: Historical Metrics

```json
{
  "schema": "dsmil-perf-history-v1",
  "binary": "llm_worker",
  "time_range": {
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-24T00:00:00Z"
  },
  "samples": 10000,
  "metrics": [
    {
      "timestamp": "2025-11-24T14:30:00Z",
      "function": "llm_decode_step",
      "invocations": 234567,
      "avg_latency_us": 873.2,
      "p50_latency_us": 801.5,
      "p99_latency_us": 1420.8,
      "cpu_cycles": 2891234,
      "cache_misses": 12847,
      "power_watts": 23.4,
      "device": "cpu",
      "actual_placement": "AMX"
    }
  ]
}
```

### 5.4 Output: Performance Forecast

```json
{
  "schema": "dsmil-perf-forecast-v1",
  "binary": "llm_worker",
  "forecast_date": "2025-11-24T15:45:00Z",
  "scenario": "production_load",
  "model": "ARIMA + Monte Carlo",
  "confidence": 0.85,
  "predictions": [
    {
      "function": "llm_decode_step",
      "current_device": "cpu_amx",
      "predicted_metrics": {
        "avg_latency_us": {
          "mean": 892.1,
          "std": 124.3,
          "p50": 853.7,
          "p99": 1502.4
        },
        "throughput_qps": {
          "mean": 227.3,
          "std": 18.4
        },
        "power_watts": {
          "mean": 24.1,
          "std": 3.2
        }
      },
      "hotspot_score": 0.87,
      "recommendation": {
        "action": "migrate_to_npu",
        "target_device": 47,
        "expected_improvement": {
          "latency_reduction": "32%",
          "power_increase": "+8W",
          "net_throughput_gain": "+45 QPS"
        },
        "confidence": 0.82
      }
    }
  ],
  "aggregate_forecast": {
    "system_qps": {
      "current": 234,
      "predicted": 279,
      "with_recommendations": 324
    },
    "power_envelope": {
      "current_avg_w": 118.3,
      "predicted_avg_w": 121.7,
      "budget_w": 120,
      "over_budget": true
    }
  },
  "alerts": [
    {
      "severity": "warning",
      "message": "Predicted power usage (121.7W) exceeds budget (120W). Consider reducing NPU utilization or implementing dynamic frequency scaling."
    }
  ]
}
```

### 5.5 Feedback Loop

```
1. Build with DSLLVM → produces *.dsmilmap
2. Deploy to production → collect runtime metrics
3. Store metrics in /var/dsmil/metrics/
4. Periodically run dsmil-ai-perf-forecast
5. Review recommendations
6. If beneficial: update source annotations or build flags
7. Rebuild with updated configuration
8. Deploy updated binary
9. Verify improvements
10. Repeat
```

---

## 6. Embedded ML Cost Models

### 6.1 `DsmilAICostModelPass`

**Purpose**: Replace heuristic cost models with ML-trained models for codegen decisions.

**Scope**:
- Inlining decisions
- Loop unrolling factors
- Vectorization strategy (scalar/SSE/AVX2/AVX-512/AMX)
- Device placement (CPU/NPU/GPU)

### 6.2 Model Format: ONNX

```
Model: dsmil_cost_model_v1.onnx
Size: ~120 MB
Input: Static code features (vector of 256 floats)
Output: Predicted speedup/penalty for each decision (vector of floats)
Inference: OpenVINO runtime on CPU/AMX/NPU
```

**Input Features** (example for vectorization decision):
- Loop trip count (static/estimated)
- Memory access patterns (stride, alignment)
- Data dependencies (RAW/WAR/WAW count)
- Arithmetic intensity (FLOPs per byte)
- Register pressure estimate
- Cache behavior hints (L1/L2/L3 miss estimates)
- Surrounding code context (embedding)

**Output**:
```
[
  speedup_scalar,        // 1.0 (baseline)
  speedup_sse,           // 1.8
  speedup_avx2,          // 3.2
  speedup_avx512,        // 4.1
  speedup_amx,           // 5.7
  speedup_npu_offload,   // 8.3 (but +latency for transfer)
  confidence             // 0.84
]
```

### 6.3 Training Pipeline

```
1. Collect training data:
   - Build 1000+ codebases with different optimization choices
   - Profile runtime performance on Meteor Lake hardware
   - Record (code_features, optimization_choice, actual_speedup)

2. Train model:
   - Use DSMIL Layer 7 infrastructure for training
   - Model: Gradient-boosted trees or small transformer
   - Loss: MSE on speedup prediction
   - Validation: 80/20 split, cross-validation

3. Export to ONNX:
   - Optimize for inference (quantization to INT8 if possible)
   - Target size: <200 MB
   - Target latency: <10ms per invocation on NPU

4. Integrate into DSLLVM:
   - Ship model with toolchain: /opt/dsmil/models/cost_model_v1.onnx
   - Load at compiler init
   - Use in DsmilAICostModelPass

5. Continuous improvement:
   - Collect feedback from production builds
   - Retrain monthly with new data
   - Version models (cost_model_v1, v2, v3, ...)
   - Allow users to select model version or provide custom models
```

### 6.4 Usage

**Automatic** (default with `--ai-mode=local`):
```bash
dsmil-clang --ai-mode=local -O3 -o output input.c
# Uses embedded cost model for all optimization decisions
```

**Custom Model**:
```bash
dsmil-clang --ai-mode=local \
  -mllvm -dsmil-cost-model=/path/to/custom_model.onnx \
  -O3 -o output input.c
```

**Disable** (use classical heuristics):
```bash
dsmil-clang --ai-mode=off -O3 -o output input.c
```

---

## 7. AI Integration Modes

### 7.1 Mode Comparison

| Mode | Local ML | External Advisors | Deterministic | Use Case |
|------|----------|-------------------|---------------|----------|
| `off` | ❌ | ❌ | ✅ | Reproducible builds, CI baseline |
| `local` | ✅ | ❌ | ✅ | Fast iterations, embedded cost models only |
| `advisor` | ✅ | ✅ | ✅* | Development with AI suggestions + validation |
| `lab` | ✅ | ✅ | ⚠️ | Experimental, may auto-apply AI suggestions |

*Deterministic after verification; AI suggestions validated by standard passes.

### 7.2 Configuration

**Via Command Line**:
```bash
dsmil-clang --ai-mode=advisor -o output input.c
```

**Via Environment Variable**:
```bash
export DSMIL_AI_MODE=local
dsmil-clang -o output input.c
```

**Via Config File** (`~/.dsmil/config.toml`):
```toml
[ai]
mode = "advisor"
local_models = "/opt/dsmil/models"
l7_advisor_url = "http://l7-llm.dsmil.internal:8080"
l8_security_url = "http://l8-security.dsmil.internal:8080"
confidence_threshold = 0.75
timeout_ms = 5000
```

---

## 8. Guardrails & Safety

### 8.1 Deterministic Verification

**Principle**: AI suggests, deterministic passes verify.

**Flow**:
```
AI Suggestion: "Set dsmil_layer=7 for function foo"
   ↓
Add to IR: !dsmil.suggested.layer = i32 7
   ↓
dsmil-layer-check pass:
   - Verify layer 7 is valid for this module
   - Check no illegal transitions introduced
   - If pass: promote to !dsmil.layer = i32 7
   - If fail: emit warning, discard suggestion
   ↓
Only verified suggestions affect final binary
```

### 8.2 Audit Logging

**Log Format**: JSON Lines
**Location**: `/var/log/dsmil/ai_advisor.jsonl`

```json
{"timestamp": "2025-11-24T15:30:45Z", "request_id": "uuid-1234", "advisor": "l7_llm", "module": "llm_inference.c", "duration_ms": 1834, "suggestions_count": 4, "applied_count": 3, "rejected_count": 1}
{"timestamp": "2025-11-24T15:30:47Z", "request_id": "uuid-1234", "suggestion": {"target": "llm_decode_step", "attr": "dsmil_layer", "value": 7, "confidence": 0.92}, "verdict": "applied", "reason": "passed layer-check validation"}
{"timestamp": "2025-11-24T15:30:47Z", "request_id": "uuid-1234", "suggestion": {"target": "llm_decode_step", "attr": "dsmil_device", "value": 999}, "verdict": "rejected", "reason": "device 999 out of range [0-103]"}
```

### 8.3 Fallback Strategy

**If AI service unavailable**:
1. Log warning: "L7 advisor unreachable, using fallback"
2. Use embedded cost models (if `--ai-mode=advisor`)
3. Use classical heuristics (if no embedded models)
4. Continue build without AI suggestions
5. Emit warning in build log

**If AI model invalid**:
1. Verify model signature (TSK-signed ONNX)
2. Check model version compatibility
3. If mismatch: fallback to last known-good model
4. Log error for ops team

### 8.4 Rate Limiting

**External Advisor Calls**:
- Max 10 requests/second per build
- Timeout: 5 seconds per request
- Retry: 2 attempts with exponential backoff
- If quota exceeded: queue or skip suggestions

**Embedded Model Inference**:
- No rate limiting (local inference)
- Watchdog: kill inference if >30 seconds
- Memory limit: 4 GB per model

---

## 9. Performance & Scaling

### 9.1 Compilation Time Impact

| Mode | Overhead | Notes |
|------|----------|-------|
| `off` | 0% | Baseline |
| `local` | 3-8% | Embedded ML inference |
| `advisor` | 10-30% | External service calls (async/parallel) |
| `lab` | 15-40% | Full AI pipeline + experimentation |

**Optimizations**:
- Parallel AI requests (multiple modules)
- Caching: reuse responses for unchanged modules
- Incremental builds: only query AI for modified code

### 9.2 AI Service Scaling

**L7 LLM Service**:
- Deployment: Kubernetes, 10 replicas
- Hardware: 10× Meteor Lake nodes (Device 47 NPU each)
- Throughput: ~100 requests/second aggregate
- Batching: group requests for efficiency

**L8 Security Service**:
- Deployment: Kubernetes, 5 replicas
- Hardware: 5× nodes with Devices 80-87
- Throughput: ~50 requests/second

### 9.3 Cost Analysis

**Per-Build AI Cost** (advisor mode):
- L7 LLM calls: ~5 requests × $0.001 = $0.005
- L8 Security calls: ~2 requests × $0.002 = $0.004
- Total: ~$0.01 per build

**Monthly Cost** (1000 builds/day):
- 30k builds × $0.01 = $300/month
- Amortized over team: negligible

---

## 10. Examples

### 10.1 Complete Flow: LLM Inference Worker

**Source** (`llm_worker.c`):
```c
#include <dsmil_attributes.h>

// No manual annotations yet; let AI suggest
void llm_decode_step(const float *input, float *output) {
    // Matrix multiply + softmax + layer norm
    matmul_kernel(input, attention_weights, output);
    softmax(output);
    layer_norm(output);
}

int main(int argc, char **argv) {
    // Process LLM requests
    return inference_loop();
}
```

**Compile**:
```bash
dsmil-clang --ai-mode=advisor \
  -fpass-pipeline=dsmil-default \
  -o llm_worker llm_worker.c
```

**AI Request** (`llm_worker.dsmilai_request.json`):
```json
{
  "schema": "dsmilai-request-v1",
  "module": {"name": "llm_worker.c"},
  "ir_summary": {
    "functions": [
      {
        "name": "llm_decode_step",
        "calls": ["matmul_kernel", "softmax", "layer_norm"],
        "memory_accesses": {"estimated_bytes": 1048576}
      }
    ]
  }
}
```

**AI Response** (`llm_worker.dsmilai_response.json`):
```json
{
  "suggestions": {
    "annotations": [
      {
        "target": "function:llm_decode_step",
        "attributes": [
          {"name": "dsmil_layer", "value": 7, "confidence": 0.92},
          {"name": "dsmil_device", "value": 47, "confidence": 0.88},
          {"name": "dsmil_stage", "value": "serve", "confidence": 0.95}
        ]
      },
      {
        "target": "function:main",
        "attributes": [
          {"name": "dsmil_sandbox", "value": "l7_llm_worker", "confidence": 0.91}
        ]
      }
    ]
  }
}
```

**DSLLVM Processing**:
1. Parse response
2. Validate suggestions (all pass)
3. Apply to IR metadata
4. Generate provenance with AI model versions
5. Link with sandbox wrapper
6. Output `llm_worker` binary + `llm_worker.dsmilmap`

**Result**: Fully annotated binary with AI-suggested (and verified) DSMIL attributes.

---

## 11. Troubleshooting

### Issue: AI service unreachable

```
error: L7 LLM advisor unreachable at http://l7-llm.dsmil.internal:8080
warning: Falling back to classical heuristics
```

**Solution**: Check network connectivity or use `--ai-mode=local`.

### Issue: Low confidence suggestions rejected

```
warning: AI suggestion for dsmil_layer=7 (confidence 0.62) below threshold (0.75), discarded
```

**Solution**: Lower threshold (`-mllvm -dsmil-ai-confidence-threshold=0.60`) or provide manual annotations.

### Issue: AI suggestion violates policy

```
error: AI suggested dsmil_layer=7 for function in layer 9 module, layer transition invalid
note: Suggestion rejected by dsmil-layer-check
```

**Solution**: AI model needs retraining or module context incomplete. Use manual annotations.

---

## 12. Future Enhancements

### 12.1 Reinforcement Learning

Train cost models using RL with real deployment feedback:
- Reward: actual speedup vs prediction
- Policy: optimization decisions
- Environment: DSMIL hardware

### 12.2 Multi-Modal AI

Combine code analysis with:
- Documentation (comments, README)
- Git history (commit messages)
- Issue tracker context

### 12.3 Continuous Learning

- Online learning: update models from production metrics
- Federated learning: aggregate across DSMIL deployments
- A/B testing: compare AI vs heuristic decisions

---

## References

1. **DSLLVM-DESIGN.md** - Main design specification
2. **DSMIL Architecture Spec** - Layer/device definitions
3. **ONNX Specification** - Model format
4. **OpenVINO Documentation** - Inference runtime

---

**End of AI Integration Guide**
