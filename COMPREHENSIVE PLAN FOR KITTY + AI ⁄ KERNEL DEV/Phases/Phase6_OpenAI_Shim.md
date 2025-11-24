# Phase 6 Supplement – OpenAI-Compatible API Shim

**Version:** 1.0
**Date:** 2025-11-23
**Status:** Implementation Ready
**Prerequisite:** Phase 6 (External API Plane), Phase 7 (L7 LLM Deployment)
**Integration:** Phase 6

---

## Executive Summary

This supplement to Phase 6 provides detailed implementation guidance for the **OpenAI-compatible API shim**, a local compatibility layer that allows existing tools (LangChain, LlamaIndex, VSCode extensions, CLI tools) to interface with DSMIL's Layer 7 LLM services without modification.

**Key Principles:**
- **Local-only access:** Bound to `127.0.0.1:8001` (not exposed externally)
- **Dumb adapter:** No policy decisions—all enforcement handled by L7 router
- **Full integration:** Respects ROE, tenant awareness, safety prompts, and hardware routing
- **Standard compliance:** Implements OpenAI API v1 spec (chat completions, completions, models)

---

## 1. Purpose & Scope

### 1.1 Problem Statement

Modern AI development tools expect OpenAI's API format:
- **LangChain/LlamaIndex:** Hardcoded to OpenAI endpoints
- **VSCode extensions:** (e.g., GitHub Copilot alternatives) Use OpenAI schema
- **CLI tools:** (e.g., `sgpt`, `shell-gpt`) Configured for OpenAI
- **Custom scripts:** Written against OpenAI SDK

**Without a shim:** Each tool requires custom integration with DSMIL's `/v1/llm` API

**With a shim:** Tools work out-of-the-box by setting:
```bash
export OPENAI_API_BASE="http://127.0.0.1:8001"
export OPENAI_API_KEY="dsmil-local-key-12345"
```

### 1.2 Scope

**In Scope:**
- OpenAI API v1 endpoints:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions` (legacy)
- Bearer token authentication
- Integration with L7 router (Device 47/48)
- Logging to SHRINK via journald

**Out of Scope:**
- External exposure (always `127.0.0.1` only)
- Streaming responses (initial implementation—can add later)
- OpenAI function calling (future enhancement)
- Embeddings endpoint (separate service if needed)
- Fine-tuning API (not applicable)

---

## 2. Architecture

### 2.1 System Context

```
┌─────────────────────────────────────────────────────────────┐
│                     Local Development Machine               │
│                                                             │
│  ┌──────────────┐          ┌─────────────────────────┐     │
│  │  LangChain   │          │   OpenAI Shim           │     │
│  │  LlamaIndex  │  HTTP    │   (127.0.0.1:8001)      │     │
│  │  VSCode Ext  │────────> │                         │     │
│  │  CLI Tools   │          │   - Auth validation     │     │
│  └──────────────┘          │   - Schema conversion   │     │
│                            │   - L7 integration      │     │
│                            └──────────┬──────────────┘     │
│                                       │                     │
│                                       │ Internal API        │
│                                       ▼                     │
│                            ┌─────────────────────────┐     │
│                            │   DSMIL L7 Router       │     │
│                            │   (Device 47/48)        │     │
│                            │                         │     │
│                            │   - ROE enforcement     │     │
│                            │   - Safety prompts      │     │
│                            │   - Tenant routing      │     │
│                            │   - Hardware selection  │     │
│                            └─────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Request Flow

1. **Client request:** LangChain sends `POST /v1/chat/completions` to `127.0.0.1:8001`
2. **Auth validation:** Shim checks `Authorization: Bearer <DSMIL_OPENAI_API_KEY>`
3. **Schema conversion:** OpenAI format → DSMIL internal format
4. **L7 invocation:** Shim calls L7 router (HTTP or direct function)
   - Passes: model/profile, messages, sampling params, tenant (if multi-tenant)
5. **L7 processing:** L7 router applies:
   - Safety prompts (prepended to system message)
   - ROE gating (if applicable)
   - Tenant-specific routing
   - Hardware selection (AMX, NPU, GPU)
6. **Response:** L7 returns structured result (text, token counts)
7. **Schema conversion:** DSMIL format → OpenAI format
8. **Client response:** Shim returns OpenAI-compliant JSON

---

## 3. API Specification

### 3.1 Service Configuration

**Service Name:** `dsmil-openai-shim`
**Bind Address:** `127.0.0.1:8001` (IPv4 loopback only)
**Protocol:** HTTP/1.1 (HTTPS not required for loopback)
**Auth:** Bearer token (`DSMIL_OPENAI_API_KEY` environment variable)

**SystemD Service File:**
```ini
[Unit]
Description=DSMIL OpenAI-Compatible API Shim
After=network.target dsmil-l7-router.service

[Service]
Type=simple
User=dsmil
Group=dsmil
Environment="DSMIL_OPENAI_API_KEY=your-secret-key-here"
Environment="DSMIL_L7_ENDPOINT=http://127.0.0.1:8007"
ExecStart=/usr/local/bin/dsmil-openai-shim
Restart=on-failure
SyslogIdentifier=dsmil-openai

[Install]
WantedBy=multi-user.target
```

### 3.2 Endpoints

#### 3.2.1 GET /v1/models

**Purpose:** List available LLM profiles

**Request:**
```http
GET /v1/models HTTP/1.1
Host: 127.0.0.1:8001
Authorization: Bearer dsmil-local-key-12345
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "dsmil-7b-amx",
      "object": "model",
      "created": 1700000000,
      "owned_by": "dsmil",
      "permission": [],
      "root": "dsmil-7b-amx",
      "parent": null
    },
    {
      "id": "dsmil-1b-npu",
      "object": "model",
      "created": 1700000000,
      "owned_by": "dsmil",
      "permission": [],
      "root": "dsmil-1b-npu",
      "parent": null
    }
  ]
}
```

**Model IDs:**
- `dsmil-7b-amx`: 7B LLM on CPU AMX (Device 47 primary)
- `dsmil-1b-npu`: 1B distilled LLM on NPU (Device 48 fallback)
- `dsmil-7b-gpu`: 7B LLM on GPU (if GPU mode enabled)
- `dsmil-instruct`: General instruction-following profile
- `dsmil-code`: Code generation profile (if available)

#### 3.2.2 POST /v1/chat/completions

**Purpose:** Chat completion (multi-turn conversation)

**Request:**
```http
POST /v1/chat/completions HTTP/1.1
Host: 127.0.0.1:8001
Authorization: Bearer dsmil-local-key-12345
Content-Type: application/json

{
  "model": "dsmil-7b-amx",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "dsmil-7b-amx",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

**Supported Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | **required** | Model ID (e.g., `dsmil-7b-amx`) |
| `messages` | array | **required** | Chat messages (role + content) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | 256 | Max tokens to generate |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | bool | false | Streaming (not implemented initially) |
| `stop` | string/array | null | Stop sequences |
| `presence_penalty` | float | 0.0 | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | 0.0 | Frequency penalty (-2.0 to 2.0) |

**Ignored Parameters (Not Supported):**
- `n` (multiple completions)
- `logit_bias`
- `user` (use for logging but not enforced)
- `functions` (function calling—future)

#### 3.2.3 POST /v1/completions

**Purpose:** Legacy text completion (single prompt)

**Request:**
```http
POST /v1/completions HTTP/1.1
Host: 127.0.0.1:8001
Authorization: Bearer dsmil-local-key-12345
Content-Type: application/json

{
  "model": "dsmil-7b-amx",
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "temperature": 0.7
}
```

**Implementation:**
Internally converted to chat format:
```python
messages = [{"role": "user", "content": prompt}]
# Then call chat completion handler
```

**Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1700000000,
  "model": "dsmil-7b-amx",
  "choices": [
    {
      "text": " Paris.\n",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 3,
    "total_tokens": 9
  }
}
```

---

## 4. Integration with L7 Router

### 4.1 L7 Router Interface

**Assumption:** L7 router exposes an internal API or Python function

**Option A: HTTP API (Recommended)**
```python
import requests

def run_l7_chat(
    profile: str,  # e.g., "dsmil-7b-amx"
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 256,
    top_p: float = 1.0,
    tenant_id: str = "LOCAL_DEV"
) -> dict:
    """
    Call L7 router via HTTP

    Returns:
        {
            "text": "The capital of France is Paris.",
            "prompt_tokens": 24,
            "completion_tokens": 8,
            "finish_reason": "stop"
        }
    """
    response = requests.post(
        "http://127.0.0.1:8007/internal/llm/chat",
        json={
            "profile": profile,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "tenant_id": tenant_id
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()
```

**Option B: Direct Function Call (If in same process)**
```python
from dsmil.l7.router import L7Router

router = L7Router()

def run_l7_chat(profile, messages, **kwargs):
    return router.generate_chat(
        profile=profile,
        messages=messages,
        **kwargs
    )
```

### 4.2 Tenant & Context Passing

**Single-Tenant Mode (Default):**
- All requests use `tenant_id = "LOCAL_DEV"`
- No ROE enforcement (development mode)

**Multi-Tenant Mode (Optional):**
- Extract tenant from API key or request header
- Pass tenant to L7 router for tenant-specific routing

**Example:**
```python
# Map API keys to tenants (stored in config or Vault)
API_KEY_TO_TENANT = {
    "dsmil-local-key-12345": "LOCAL_DEV",
    "dsmil-alpha-key-67890": "ALPHA",
    "dsmil-bravo-key-abcde": "BRAVO"
}

def get_tenant_from_api_key(api_key: str) -> str:
    return API_KEY_TO_TENANT.get(api_key, "LOCAL_DEV")
```

### 4.3 Safety Prompts & ROE Integration

**Shim does NOT apply safety prompts**—this is L7's responsibility.

L7 router should:
1. Receive messages from shim
2. Prepend safety system message (if configured):
   ```
   "You are a helpful, harmless, and honest AI assistant.
    Do not generate harmful, illegal, or offensive content."
   ```
3. Check ROE token (if tenant requires it)
4. Route to appropriate hardware (AMX/NPU/GPU)
5. Generate response
6. Return to shim

**This ensures:**
- Shim remains dumb (no policy logic)
- All enforcement is centralized in L7
- Consistency across all L7 access methods (API, shim, internal)

---

## 5. Implementation Guide

### 5.1 Technology Stack

**Recommended:**
- **Framework:** FastAPI (Python) or Express (Node.js)
- **Why:** Lightweight, easy OpenAPI integration, async support
- **Auth:** Simple bearer token check (no OAuth complexity)
- **Logging:** Python `logging` → journald with `SyslogIdentifier=dsmil-openai`

### 5.2 Python Implementation Sketch

**File:** `dsmil_openai_shim.py`

```python
#!/usr/bin/env python3
"""DSMIL OpenAI-Compatible API Shim"""

import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests

# Configuration
DSMIL_OPENAI_API_KEY = os.getenv("DSMIL_OPENAI_API_KEY", "dsmil-default-key")
DSMIL_L7_ENDPOINT = os.getenv("DSMIL_L7_ENDPOINT", "http://127.0.0.1:8007")

app = FastAPI(title="DSMIL OpenAI Shim", version="1.0.0")
security = HTTPBearer()

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

# Auth
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != DSMIL_OPENAI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Endpoints
@app.get("/v1/models")
def list_models(token: str = Security(verify_token)):
    """List available models"""
    return {
        "object": "list",
        "data": [
            {"id": "dsmil-7b-amx", "object": "model", "created": 1700000000, "owned_by": "dsmil"},
            {"id": "dsmil-1b-npu", "object": "model", "created": 1700000000, "owned_by": "dsmil"},
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, token: str = Security(verify_token)):
    """Chat completion endpoint"""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")

    # Convert to L7 format
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Call L7 router
    try:
        l7_response = requests.post(
            f"{DSMIL_L7_ENDPOINT}/internal/llm/chat",
            json={
                "profile": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "tenant_id": "LOCAL_DEV"
            },
            timeout=30
        ).json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"L7 error: {str(e)}")

    # Convert to OpenAI format
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": l7_response["text"]
                },
                "finish_reason": l7_response.get("finish_reason", "stop")
            }
        ],
        "usage": {
            "prompt_tokens": l7_response.get("prompt_tokens", 0),
            "completion_tokens": l7_response.get("completion_tokens", 0),
            "total_tokens": l7_response.get("prompt_tokens", 0) + l7_response.get("completion_tokens", 0)
        }
    }

@app.post("/v1/completions")
def completions(request: CompletionRequest, token: str = Security(verify_token)):
    """Legacy text completion endpoint"""
    # Convert to chat format
    messages = [{"role": "user", "content": request.prompt}]
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=[ChatMessage(role="user", content=request.prompt)],
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Reuse chat handler
    chat_response = chat_completions(chat_request, token)

    # Convert to completion format
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": chat_response["created"],
        "model": request.model,
        "choices": [
            {
                "text": chat_response["choices"][0]["message"]["content"],
                "index": 0,
                "logprobs": None,
                "finish_reason": chat_response["choices"][0]["finish_reason"]
            }
        ],
        "usage": chat_response["usage"]
    }

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_config={
        "version": 1,
        "handlers": {
            "default": {
                "class": "logging.handlers.SysLogHandler",
                "address": "/dev/log",
                "ident": "dsmil-openai"
            }
        }
    })
```

### 5.3 Deployment Steps

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn pydantic requests
   ```

2. **Configure environment:**
   ```bash
   export DSMIL_OPENAI_API_KEY="your-secret-key-here"
   export DSMIL_L7_ENDPOINT="http://127.0.0.1:8007"
   ```

3. **Run shim:**
   ```bash
   python dsmil_openai_shim.py
   ```

4. **Test:**
   ```bash
   curl -X POST http://127.0.0.1:8001/v1/chat/completions \
     -H "Authorization: Bearer your-secret-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "dsmil-7b-amx",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'
   ```

5. **Configure tools:**
   ```bash
   # LangChain
   export OPENAI_API_BASE="http://127.0.0.1:8001"
   export OPENAI_API_KEY="your-secret-key-here"

   # LlamaIndex
   export OPENAI_API_BASE="http://127.0.0.1:8001"
   export OPENAI_API_KEY="your-secret-key-here"
   ```

---

## 6. Logging & Observability

### 6.1 Logging Strategy

**All requests logged with:**
- Request ID (correlation)
- Model requested
- Prompt length (tokens)
- Response length (tokens)
- Latency (ms)
- Tenant ID (if multi-tenant)
- Error messages (if failed)

**Log Destination:**
- `SyslogIdentifier=dsmil-openai`
- Aggregated to `/var/log/dsmil.log` via journald
- Ingested by Loki → SHRINK dashboard

**Example Log:**
```
2025-11-23T12:34:56Z dsmil-openai[1234]: request_id=chatcmpl-abc123 model=dsmil-7b-amx tenant=LOCAL_DEV prompt_tokens=24 completion_tokens=8 latency_ms=1850 status=success
```

### 6.2 Metrics (Prometheus)

**Metrics to Export:**
| Metric | Type | Description |
|--------|------|-------------|
| `dsmil_openai_requests_total` | Counter | Total requests by model and status |
| `dsmil_openai_latency_seconds` | Histogram | Request latency distribution |
| `dsmil_openai_prompt_tokens_total` | Counter | Total prompt tokens processed |
| `dsmil_openai_completion_tokens_total` | Counter | Total completion tokens generated |
| `dsmil_openai_errors_total` | Counter | Total errors by type |

**Integration:**
```python
from prometheus_client import Counter, Histogram, generate_latest

requests_total = Counter('dsmil_openai_requests_total', 'Total requests', ['model', 'status'])
latency = Histogram('dsmil_openai_latency_seconds', 'Request latency')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## 7. Testing & Validation

### 7.1 Integration Tests

**Test Cases:**

1. **Authentication:**
   - ✅ Valid API key → 200 OK
   - ✅ Invalid API key → 401 Unauthorized
   - ✅ Missing Authorization header → 401 Unauthorized

2. **Models Endpoint:**
   - ✅ GET /v1/models returns list of models
   - ✅ Model IDs match expected (dsmil-7b-amx, etc.)

3. **Chat Completions:**
   - ✅ Simple user message → valid response
   - ✅ Multi-turn conversation → context maintained
   - ✅ Temperature/max_tokens respected
   - ✅ Stop sequences work
   - ✅ Error handling (L7 timeout, invalid model)

4. **Text Completions:**
   - ✅ Legacy prompt format → valid response
   - ✅ Conversion to chat format correct

5. **L7 Integration:**
   - ✅ Shim calls L7 router correctly
   - ✅ Tenant passed through
   - ✅ Safety prompts applied by L7 (not shim)
   - ✅ ROE enforcement works (if enabled)

6. **Observability:**
   - ✅ Logs appear in journald with correct identifier
   - ✅ Prometheus metrics exported
   - ✅ SHRINK dashboard shows traffic

**Test Script:**
```bash
#!/bin/bash
# test_openai_shim.sh

BASE_URL="http://127.0.0.1:8001"
API_KEY="your-secret-key-here"

# Test 1: List models
echo "Test 1: List models"
curl -X GET "$BASE_URL/v1/models" \
  -H "Authorization: Bearer $API_KEY"

# Test 2: Chat completion
echo "\nTest 2: Chat completion"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dsmil-7b-amx",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'

# Test 3: Invalid auth
echo "\nTest 3: Invalid auth (should fail)"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer wrong-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "dsmil-7b-amx", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 8. Security Considerations

### 8.1 Threat Model

**Mitigated Threats:**
- **Unauthorized access:** API key required (local-only reduces exposure)
- **External exposure:** Bound to 127.0.0.1 (not reachable from network)
- **Injection attacks:** Input validation via Pydantic schemas

**Residual Risks:**
- **API key theft:** If key leaked, attacker with local access can use LLM
  - **Mitigation:** Rotate key regularly, monitor usage for anomalies
- **Local privilege escalation:** Attacker with local shell can access shim
  - **Mitigation:** Run shim as non-root user, file permissions on config

### 8.2 Best Practices

1. **API Key Management:**
   - Store in environment variable or Vault (not in code)
   - Rotate quarterly
   - Use separate keys for dev/staging/prod (if applicable)

2. **Logging:**
   - Do NOT log API keys or full prompts (PII/sensitive data)
   - Log request IDs for correlation
   - Sanitize error messages (no stack traces to user)

3. **Rate Limiting (Optional):**
   - Add per-key rate limit (e.g., 100 req/min) to prevent abuse
   - Use `slowapi` or similar library

4. **Monitoring:**
   - Alert on unusual patterns (e.g., 1000 requests in 1 min from single key)
   - SHRINK dashboard should show shim traffic separately

---

## 9. Completion Criteria

Phase 6 (with OpenAI Shim) is complete when:

- ✅ External `/v1/*` DSMIL API is live (Phase 6 core)
- ✅ OpenAI shim running on `127.0.0.1:8001`
- ✅ `/v1/models`, `/v1/chat/completions`, `/v1/completions` implemented
- ✅ `DSMIL_OPENAI_API_KEY` enforced
- ✅ Shim integrates with L7 router (respects ROE, safety prompts, tenant routing)
- ✅ All requests logged to `/var/log/dsmil.log` with `SyslogIdentifier=dsmil-openai`
- ✅ SHRINK displays shim traffic and anomalies
- ✅ Integration tests pass (auth, models, chat, completions)
- ✅ LangChain/LlamaIndex/CLI tools work with shim (validated manually)

---

## 10. Future Enhancements (Post-MVP)

1. **Streaming Support:**
   - Implement Server-Sent Events (SSE) for `stream=true`
   - Useful for interactive chat UIs

2. **Function Calling:**
   - Add OpenAI function calling support
   - Map to DSMIL tool-use capabilities (if available)

3. **Embeddings Endpoint:**
   - `POST /v1/embeddings` for vector generation
   - Integrate with Layer 6 retrieval (if applicable)

4. **Multi-Tenant API Keys:**
   - Map different API keys to different tenants
   - Enable per-tenant usage tracking and quotas

5. **OpenAI SDK Compatibility:**
   - Test with official OpenAI Python SDK
   - Ensure full compatibility with SDK features

---

## 11. Metadata

**Author:** DSMIL Implementation Team
**Integration Phase:** Phase 6 (External API Plane)
**Dependencies:**
- Phase 6 core (External API)
- Phase 7 (Layer 7 LLM operational)
- L7 router with internal API

**Version History:**
- v1.0 (2025-11-23): Initial specification (based on Phase7a.txt notes)

---

**End of OpenAI Shim Specification**

**Next:** If you want, I can provide a concrete `run_l7_chat()` implementation sketch that calls your L7 router (e.g., via HTTP) and passes through tenant/context so the shim remains purely an adapter.
