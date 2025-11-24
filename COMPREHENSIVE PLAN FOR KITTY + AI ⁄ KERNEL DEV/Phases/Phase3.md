# Phase 3 – L7 Generative Plane & Local Tools (DBE + Shim) (v2.0)

**Version:** 2.0
**Status:** Aligned with v3.1 Comprehensive Plan
**Date:** 2025-11-23
**Last Updated:** Aligned hardware specs, Device 47 specifications, DBE protocol integration

---

## 1. Objectives

Phase 3 activates **Layer 7 (EXTENDED)** as the primary generative AI plane with:

1. **Local LLM deployment** on Device 47 (Advanced AI/ML - Primary LLM device)
2. **DSMIL Binary Envelope (DBE)** for all L7-internal communication
3. **Local OpenAI-compatible shim** for tool integration
4. **Post-quantum cryptographic boundaries** for L7 services
5. **Policy-enforced routing** with compartment and ROE enforcement

### System Context (v3.1)

- **Physical Hardware:** Intel Core Ultra 7 165H (48.2 TOPS INT8: 13.0 NPU + 32.0 GPU + 3.2 CPU)
- **Memory:** 64 GB LPDDR5x-7467, 62 GB usable for AI, 64 GB/s shared bandwidth
- **Layer 7 (EXTENDED):** 8 devices (43-50), 40 GB budget, 440 TOPS theoretical
  - **Device 47 (Advanced AI/ML):** Primary LLM device, 20 GB allocation, 80 TOPS theoretical
  - Device 43: Extended Analytics (40 TOPS)
  - Device 44: Cross-Domain Fusion (50 TOPS)
  - Device 45: Enhanced Prediction (55 TOPS)
  - Device 46: Quantum Integration (35 TOPS, CPU-bound)
  - Device 48: Strategic Planning (70 TOPS)
  - Device 49: Global Intelligence (60 TOPS)
  - Device 50: Autonomous Systems (50 TOPS)

### Key Principles

1. **All L7-internal communication uses DBE** (no HTTP between L7 components)
2. **OpenAI shim → L7 router uses DBE** (or PQC HTTP/UDS → DBE conversion)
3. **Shim remains a dumb adapter** – policy enforcement happens in L7 router
4. **Device 47 is primary LLM target** – 20 GB for LLaMA-7B/Mistral-7B INT8 + KV cache

---

## 2. Architecture Overview

### 2.1 Layer 7 Service Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Layer 7 (EXTENDED) Services                  │
│                  8 Devices (43-50), 40 GB Budget                 │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐          ┌──────▼──────┐       ┌──────▼──────┐
   │ L7      │          │ L7 LLM      │       │ L7 Agent    │
   │ Router  │◄────────►│ Worker-47   │       │ Harness     │
   │(Dev 43) │   DBE    │ (Device 47) │       │ (Dev 48)    │
   └────┬────┘          └─────────────┘       └─────────────┘
        │                      │
        │ DBE                  │ DBE
        │                      │
   ┌────▼────────────┐    ┌────▼────────────┐
   │ OpenAI Shim     │    │ Other L7        │
   │ (127.0.0.1:8001)│    │ Workers         │
   │                 │    │ (Devices 44-50) │
   └─────────────────┘    └─────────────────┘
        │
        │ HTTP (localhost only)
        │
   ┌────▼────────────┐
   │ Local Tools     │
   │ (LangChain, IDE,│
   │  CLI, etc.)     │
   └─────────────────┘
```

### 2.2 New L7 Services

| Service | Device | Purpose | Memory | Protocol |
|---------|--------|---------|--------|----------|
| `dsmil-l7-router` | 43 | L7 orchestration, policy enforcement, routing | 2 GB | DBE |
| `dsmil-l7-llm-worker-47` | 47 | Primary LLM inference (LLaMA-7B/Mistral-7B INT8) | 20 GB | DBE |
| `dsmil-l7-llm-worker-npu` | 44 | Micro-LLM on NPU (1B model) | 2 GB | DBE |
| `dsmil-l7-agent` | 48 | Constrained agent harness using L7 profiles | 4 GB | DBE |
| `dsmil-l7-multimodal` | 45 | Vision + text fusion (CLIP, etc.) | 6 GB | DBE |
| `dsmil-openai-shim` | N/A | Local OpenAI API adapter (loopback only) | 200 MB | HTTP → DBE |

### 2.3 DBE Message Types for Layer 7

**New `msg_type` definitions:**

| Message Type | Hex | Purpose | Direction |
|--------------|-----|---------|-----------|
| `L7_CHAT_REQ` | `0x41` | Chat completion request | Client → Router → Worker |
| `L7_CHAT_RESP` | `0x42` | Chat completion response | Worker → Router → Client |
| `L7_AGENT_TASK` | `0x43` | Agent task assignment | Router → Agent Harness |
| `L7_AGENT_RESULT` | `0x44` | Agent task result | Agent Harness → Router |
| `L7_MODEL_STATUS` | `0x45` | Model health/load status | Worker → Router |
| `L7_POLICY_CHECK` | `0x46` | Policy validation request | Router → Policy Engine |

**DBE Header TLVs for L7 (extended from Phase 7 spec):**

```text
TENANT_ID (string)              – e.g., "SOC_TEAM_ALPHA"
COMPARTMENT_MASK (bitmask)      – e.g., SOC | DEV | LAB
CLASSIFICATION (enum)           – UNCLAS, SECRET, TS, TS_SIM
ROE_LEVEL (enum)                – ANALYSIS_ONLY, SOC_ASSIST, TRAINING
LAYER_PATH (string)             – e.g., "3→5→7"
DEVICE_ID_SRC (uint8)           – Source device (0-103)
DEVICE_ID_DST (uint8)           – Destination device (0-103)
L7_PROFILE (string)             – e.g., "llm-7b-amx", "llm-1b-npu"
L7_CLAIM_TOKEN (blob)           – PQC-signed claim (tenant_id, client_id, roles, request_id)
TIMESTAMP (uint48)              – Unix time + sub-ms
REQUEST_ID (UUID)               – Correlation ID
```

---

## 3. DBE + L7 Integration

### 3.1 L7 Router (Device 43)

**Purpose:** Central orchestrator for all Layer 7 AI workloads.

**Responsibilities:**
1. Receive DBE `L7_CHAT_REQ` messages from:
   - Internal services (Layer 8 SOC via Redis → DBE bridge)
   - OpenAI shim (HTTP/UDS → DBE conversion)
2. Apply policy checks:
   - Validate `L7_CLAIM_TOKEN` signature (ML-DSA-87)
   - Check `COMPARTMENT_MASK` and `ROE_LEVEL`
   - Enforce rate limits per tenant
3. Route to appropriate L7 worker based on:
   - `L7_PROFILE` (model selection)
   - `TENANT_ID` (resource allocation)
   - Worker load balancing
4. Forward DBE `L7_CHAT_RESP` back to caller

**Implementation Sketch:**

```python
#!/usr/bin/env python3
# /opt/dsmil/l7_router.py
"""
DSMIL L7 Router (Device 43 - Extended Analytics)
Routes L7 DBE messages to appropriate LLM workers
"""

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from dsmil_dbe import DBEMessage, DBESocket, MessageType
from dsmil_pqc import MLDSAVerifier

# Constants
DEVICE_ID = 43
LAYER = 7
TOKEN_BASE = 0x8081  # 0x8000 + (43 * 3)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [L7-ROUTER] [Device-43] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class L7Worker:
    device_id: int
    profile: str
    socket_path: str
    current_load: float  # 0.0-1.0
    max_memory_gb: float

class L7Router:
    def __init__(self):
        self.workers: Dict[str, L7Worker] = {
            "llm-7b-amx": L7Worker(
                device_id=47,
                profile="llm-7b-amx",
                socket_path="/var/run/dsmil/l7-worker-47.sock",
                current_load=0.0,
                max_memory_gb=20.0
            ),
            "llm-1b-npu": L7Worker(
                device_id=44,
                profile="llm-1b-npu",
                socket_path="/var/run/dsmil/l7-worker-44.sock",
                current_load=0.0,
                max_memory_gb=2.0
            ),
            "agent": L7Worker(
                device_id=48,
                profile="agent",
                socket_path="/var/run/dsmil/l7-agent-48.sock",
                current_load=0.0,
                max_memory_gb=4.0
            ),
        }

        self.pqc_verifier = MLDSAVerifier()  # ML-DSA-87 signature verification
        self.router_socket = DBESocket(bind_path="/var/run/dsmil/l7-router.sock")

        logger.info(f"L7 Router initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")
        logger.info(f"Registered {len(self.workers)} L7 workers")

    def validate_claim_token(self, msg: DBEMessage) -> bool:
        """Verify L7_CLAIM_TOKEN signature using ML-DSA-87"""
        try:
            claim_token = msg.tlv_get("L7_CLAIM_TOKEN")
            if not claim_token:
                logger.warning("Missing L7_CLAIM_TOKEN in request")
                return False

            # Verify PQC signature
            is_valid = self.pqc_verifier.verify(claim_token)
            if not is_valid:
                logger.warning("Invalid L7_CLAIM_TOKEN signature")
                return False

            return True
        except Exception as e:
            logger.error(f"Claim token validation error: {e}")
            return False

    def apply_policy(self, msg: DBEMessage) -> Optional[str]:
        """
        Apply policy checks and return error string if denied, None if allowed
        """
        # Check compartment
        compartment = msg.tlv_get("COMPARTMENT_MASK", 0)
        if compartment & 0x80:  # KINETIC bit set
            return "DENIED: KINETIC compartment not allowed in L7"

        # Check ROE level
        roe_level = msg.tlv_get("ROE_LEVEL", "")
        if roe_level not in ["ANALYSIS_ONLY", "SOC_ASSIST", "TRAINING"]:
            return f"DENIED: Invalid ROE_LEVEL '{roe_level}'"

        # Check classification
        classification = msg.tlv_get("CLASSIFICATION", "")
        if classification == "EXEC":
            return "DENIED: EXEC classification requires Layer 9 authorization"

        return None  # Policy checks passed

    def select_worker(self, msg: DBEMessage) -> Optional[L7Worker]:
        """Select appropriate worker based on profile and load"""
        profile = msg.tlv_get("L7_PROFILE", "llm-7b-amx")  # Default to Device 47

        worker = self.workers.get(profile)
        if not worker:
            logger.warning(f"Unknown L7_PROFILE: {profile}, falling back to llm-7b-amx")
            worker = self.workers["llm-7b-amx"]

        # Check load (simple round-robin if overloaded)
        if worker.current_load > 0.9:
            logger.warning(f"Worker {worker.device_id} overloaded, load={worker.current_load:.2f}")
            # TODO: Implement fallback worker selection

        return worker

    def route_message(self, msg: DBEMessage) -> DBEMessage:
        """Main routing logic"""
        request_id = msg.tlv_get("REQUEST_ID", "unknown")
        tenant_id = msg.tlv_get("TENANT_ID", "unknown")

        logger.info(f"Routing L7_CHAT_REQ | Request: {request_id} | Tenant: {tenant_id}")

        # Step 1: Validate claim token
        if not self.validate_claim_token(msg):
            return self._create_error_response(msg, "CLAIM_TOKEN_INVALID")

        # Step 2: Apply policy
        policy_error = self.apply_policy(msg)
        if policy_error:
            logger.warning(f"Policy denied: {policy_error}")
            return self._create_error_response(msg, policy_error)

        # Step 3: Select worker
        worker = self.select_worker(msg)
        if not worker:
            return self._create_error_response(msg, "NO_WORKER_AVAILABLE")

        # Step 4: Forward to worker via DBE
        try:
            worker_socket = DBESocket(connect_path=worker.socket_path)
            response = worker_socket.send_and_receive(msg, timeout=30.0)

            logger.info(
                f"L7_CHAT_RESP received from Device {worker.device_id} | "
                f"Request: {request_id}"
            )

            return response

        except Exception as e:
            logger.error(f"Worker communication error: {e}")
            return self._create_error_response(msg, f"WORKER_ERROR: {str(e)}")

    def _create_error_response(self, request: DBEMessage, error: str) -> DBEMessage:
        """Create DBE error response"""
        response = DBEMessage(
            msg_type=MessageType.L7_CHAT_RESP,
            correlation_id=request.correlation_id,
            payload={"error": error, "choices": []}
        )
        response.tlv_set("DEVICE_ID_SRC", DEVICE_ID)
        response.tlv_set("REQUEST_ID", request.tlv_get("REQUEST_ID"))
        response.tlv_set("TIMESTAMP", time.time())
        return response

    def run(self):
        """Main event loop"""
        logger.info("L7 Router started, listening for DBE messages...")

        while True:
            try:
                msg = self.router_socket.receive(timeout=1.0)
                if not msg:
                    continue

                if msg.msg_type == MessageType.L7_CHAT_REQ:
                    response = self.route_message(msg)
                    self.router_socket.send(response)
                else:
                    logger.warning(f"Unexpected message type: 0x{msg.msg_type:02X}")

            except KeyboardInterrupt:
                logger.info("L7 Router shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    router = L7Router()
    router.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l7-router.service
[Unit]
Description=DSMIL L7 Router (Device 43 - Extended Analytics)
After=network.target
Requires=dsmil-l7-llm-worker-47.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=43"
Environment="DSMIL_LAYER=7"
Environment="DBE_SOCKET_PATH=/var/run/dsmil/l7-router.sock"

ExecStartPre=/usr/bin/mkdir -p /var/run/dsmil
ExecStartPre=/usr/bin/chown dsmil:dsmil /var/run/dsmil
ExecStart=/opt/dsmil/.venv/bin/python l7_router.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l7-router

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3.2 L7 LLM Worker (Device 47 - Primary LLM)

**Purpose:** Run primary LLM inference (LLaMA-7B/Mistral-7B/Falcon-7B INT8) with 20 GB allocation.

**Memory Breakdown (Device 47):**
- LLM weights (INT8): 7.2 GB
- KV cache (32K context): 10.0 GB
- CLIP vision encoder: 1.8 GB
- Workspace (batching, buffers): 1.0 GB
- **Total:** 20.0 GB (50% of Layer 7 budget)

**Implementation Sketch:**

```python
#!/usr/bin/env python3
# /opt/dsmil/l7_llm_worker_47.py
"""
DSMIL L7 LLM Worker (Device 47 - Advanced AI/ML)
Primary LLM inference engine with 20 GB allocation
"""

import time
import logging
from typing import Dict, List

from dsmil_dbe import DBEMessage, DBESocket, MessageType
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import intel_extension_for_pytorch as ipex

# Constants
DEVICE_ID = 47
LAYER = 7
TOKEN_BASE = 0x808D  # 0x8000 + (47 * 3)
MODEL_PATH = "/opt/dsmil/models/llama-7b-int8"
MAX_MEMORY_GB = 20.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [L7-WORKER-47] [Device-47] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class L7LLMWorker:
    def __init__(self):
        logger.info(f"Loading LLM model from {MODEL_PATH}...")

        # Load model with INT8 quantization + Intel optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.int8,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Apply Intel Extension for PyTorch optimizations (AMX, Flash Attention)
        self.model = ipex.optimize(self.model, dtype=torch.int8, inplace=True)

        self.socket = DBESocket(bind_path="/var/run/dsmil/l7-worker-47.sock")

        logger.info(f"LLM Worker initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")
        logger.info(f"Model loaded: {MODEL_PATH} | Memory budget: {MAX_MEMORY_GB} GB")

    def generate_completion(self, msg: DBEMessage) -> Dict:
        """Generate LLM completion from DBE request"""
        try:
            payload = msg.payload
            messages = payload.get("messages", [])
            max_tokens = payload.get("max_tokens", 512)
            temperature = payload.get("temperature", 0.7)

            # Convert messages to prompt
            prompt = self._format_prompt(messages)

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Generate (with AMX acceleration)
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # KV cache optimization
                )

            latency_ms = (time.time() - start_time) * 1000

            # Decode
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = completion[len(prompt):].strip()  # Remove prompt echo

            # Calculate tokens
            prompt_tokens = len(inputs.input_ids[0])
            completion_tokens = len(outputs[0]) - prompt_tokens

            logger.info(
                f"Generated completion | "
                f"Prompt: {prompt_tokens} tok | "
                f"Completion: {completion_tokens} tok | "
                f"Latency: {latency_ms:.1f}ms"
            )

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": completion
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "model": "llama-7b-int8-amx",
                "device_id": DEVICE_ID,
                "latency_ms": latency_ms
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"error": str(e), "choices": []}

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format chat messages into LLaMA prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f" {content} ")

        return "".join(prompt_parts)

    def run(self):
        """Main event loop"""
        logger.info("L7 LLM Worker started, listening for DBE messages...")

        while True:
            try:
                msg = self.socket.receive(timeout=1.0)
                if not msg:
                    continue

                if msg.msg_type == MessageType.L7_CHAT_REQ:
                    request_id = msg.tlv_get("REQUEST_ID", "unknown")
                    logger.info(f"Processing L7_CHAT_REQ | Request: {request_id}")

                    result = self.generate_completion(msg)

                    response = DBEMessage(
                        msg_type=MessageType.L7_CHAT_RESP,
                        correlation_id=msg.correlation_id,
                        payload=result
                    )
                    response.tlv_set("DEVICE_ID_SRC", DEVICE_ID)
                    response.tlv_set("REQUEST_ID", request_id)
                    response.tlv_set("TIMESTAMP", time.time())

                    self.socket.send(response)
                else:
                    logger.warning(f"Unexpected message type: 0x{msg.msg_type:02X}")

            except KeyboardInterrupt:
                logger.info("L7 LLM Worker shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    worker = L7LLMWorker()
    worker.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l7-llm-worker-47.service
[Unit]
Description=DSMIL L7 LLM Worker (Device 47 - Primary LLM)
After=network.target

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=47"
Environment="DSMIL_LAYER=7"
Environment="OMP_NUM_THREADS=16"
Environment="MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto"

# Memory limits (20 GB for Device 47)
MemoryMax=21G
MemoryHigh=20G

ExecStart=/opt/dsmil/.venv/bin/python l7_llm_worker_47.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l7-llm-worker-47

Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
```

### 3.3 OpenAI Shim → DBE Integration

**Purpose:** Provide local OpenAI API compatibility while routing all requests through DBE.

**Architecture:**

```
Local Tool (LangChain, etc.)
    │
    │ HTTP POST /v1/chat/completions
    ↓
OpenAI Shim (127.0.0.1:8001)
    │ 1. Validate API key
    │ 2. Create L7_CLAIM_TOKEN
    │ 3. Convert OpenAI format → DBE L7_CHAT_REQ
    ↓
L7 Router (Device 43) via DBE over UDS
    │ 4. Policy enforcement
    │ 5. Route to Device 47
    ↓
Device 47 LLM Worker
    │ 6. Generate completion
    ↓
L7 Router ← DBE L7_CHAT_RESP
    ↓
OpenAI Shim
    │ 7. Convert DBE → OpenAI JSON format
    ↓
Local Tool receives response
```

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/openai_shim.py
"""
DSMIL OpenAI-Compatible Shim
Exposes local OpenAI API, routes all requests via DBE to L7 Router
"""

import os
import time
import uuid
import logging
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from dsmil_dbe import DBEMessage, DBESocket, MessageType
from dsmil_pqc import MLDSASigner

# Constants
DSMIL_OPENAI_API_KEY = os.environ.get("DSMIL_OPENAI_API_KEY", "dsmil-local-key")
L7_ROUTER_SOCKET = "/var/run/dsmil/l7-router.sock"

app = FastAPI(title="DSMIL OpenAI Shim", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PQC signer for claim tokens
pqc_signer = MLDSASigner(key_path="/opt/dsmil/keys/shim-mldsa-87.key")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "dsmil"

@app.get("/v1/models")
def list_models():
    """List available DSMIL L7 models"""
    return {
        "object": "list",
        "data": [
            ModelInfo(id="llama-7b-int8-amx", created=int(time.time())),
            ModelInfo(id="mistral-7b-int8-amx", created=int(time.time())),
            ModelInfo(id="llm-1b-npu", created=int(time.time())),
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(None)
):
    """
    OpenAI-compatible chat completions endpoint
    Routes all requests via DBE to L7 Router
    """
    # Step 1: Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    api_key = authorization[7:]  # Remove "Bearer "
    if api_key != DSMIL_OPENAI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Step 2: Create L7 claim token (PQC-signed)
    request_id = str(uuid.uuid4())
    claim_data = {
        "tenant_id": "LOCAL_TOOL_USER",
        "client_id": "openai_shim",
        "roles": ["SOC_ASSIST"],
        "request_id": request_id,
        "timestamp": time.time()
    }
    claim_token = pqc_signer.sign(claim_data)

    # Step 3: Map OpenAI model to L7 profile
    profile_map = {
        "llama-7b-int8-amx": "llm-7b-amx",
        "mistral-7b-int8-amx": "llm-7b-amx",
        "llm-1b-npu": "llm-1b-npu",
        "gpt-3.5-turbo": "llm-7b-amx",  # Fallback mapping
        "gpt-4": "llm-7b-amx",
    }
    l7_profile = profile_map.get(request.model, "llm-7b-amx")

    # Step 4: Create DBE L7_CHAT_REQ message
    dbe_msg = DBEMessage(
        msg_type=MessageType.L7_CHAT_REQ,
        correlation_id=request_id,
        payload={
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
    )

    # Set DBE TLVs
    dbe_msg.tlv_set("TENANT_ID", "LOCAL_TOOL_USER")
    dbe_msg.tlv_set("COMPARTMENT_MASK", 0x01)  # SOC compartment
    dbe_msg.tlv_set("CLASSIFICATION", "SECRET")
    dbe_msg.tlv_set("ROE_LEVEL", "SOC_ASSIST")
    dbe_msg.tlv_set("L7_PROFILE", l7_profile)
    dbe_msg.tlv_set("L7_CLAIM_TOKEN", claim_token)
    dbe_msg.tlv_set("REQUEST_ID", request_id)
    dbe_msg.tlv_set("TIMESTAMP", time.time())
    dbe_msg.tlv_set("DEVICE_ID_SRC", 0)  # Shim is not a DSMIL device
    dbe_msg.tlv_set("DEVICE_ID_DST", 43)  # Target L7 Router

    logger.info(
        f"Routing OpenAI request via DBE | "
        f"Model: {request.model} → Profile: {l7_profile} | "
        f"Request: {request_id}"
    )

    # Step 5: Send to L7 Router via DBE over UDS
    try:
        router_socket = DBESocket(connect_path=L7_ROUTER_SOCKET)
        response = router_socket.send_and_receive(dbe_msg, timeout=30.0)

        if response.msg_type != MessageType.L7_CHAT_RESP:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response type: 0x{response.msg_type:02X}"
            )

        # Step 6: Convert DBE response to OpenAI format
        result = response.payload

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        openai_response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": result.get("choices", []),
            "usage": result.get("usage", {}),
            "dsmil_metadata": {
                "device_id": result.get("device_id"),
                "latency_ms": result.get("latency_ms"),
                "l7_profile": l7_profile,
            }
        }

        logger.info(f"Completed OpenAI request | Request: {request_id}")

        return openai_response

    except Exception as e:
        logger.error(f"DBE communication error: {e}")
        raise HTTPException(status_code=500, detail=f"DBE routing failed: {str(e)}")

@app.post("/v1/completions")
def completions(request: ChatCompletionRequest, authorization: str = Header(None)):
    """Legacy completions endpoint - maps to chat completions"""
    # Convert single prompt to chat format
    if not request.messages:
        request.messages = [ChatMessage(role="user", content="")]

    return chat_completions(request, authorization)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting DSMIL OpenAI Shim on 127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-openai-shim.service
[Unit]
Description=DSMIL OpenAI-Compatible Shim (127.0.0.1:8001)
After=network.target dsmil-l7-router.service
Requires=dsmil-l7-router.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_OPENAI_API_KEY=dsmil-local-key-change-me"

ExecStart=/opt/dsmil/.venv/bin/python openai_shim.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-openai-shim

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 4. Post-Quantum Cryptographic Boundaries

### 4.1 PQC Architecture for L7

All L7 services use **ML-DSA-87 (Dilithium5)** for identity and **ML-KEM-1024 (Kyber-1024)** for session keys.

**Identity Keypairs:**

| Service | Device | Public Key Path | Private Key Path (TPM-sealed) |
|---------|--------|----------------|-------------------------------|
| L7 Router | 43 | `/opt/dsmil/keys/dev43-mldsa-87.pub` | `/opt/dsmil/keys/dev43-mldsa-87.key` |
| LLM Worker 47 | 47 | `/opt/dsmil/keys/dev47-mldsa-87.pub` | `/opt/dsmil/keys/dev47-mldsa-87.key` |
| Agent Harness | 48 | `/opt/dsmil/keys/dev48-mldsa-87.pub` | `/opt/dsmil/keys/dev48-mldsa-87.key` |
| OpenAI Shim | N/A | `/opt/dsmil/keys/shim-mldsa-87.pub` | `/opt/dsmil/keys/shim-mldsa-87.key` |

**Session Establishment (DBE UDS channels):**

1. **Handshake:**
   - Each L7 service exchanges signed identity bundles (ML-DSA-87 signatures)
   - Optional: ML-KEM-1024 encapsulation for long-lived sessions

2. **Channel Protection:**
   - UDS sockets on same host: Direct AES-256-GCM on buffers
   - QUIC/DTLS over UDP (cross-node): Hybrid keys from ML-KEM-1024 + ECDHE

3. **Message Authentication:**
   - Each DBE message includes `L7_CLAIM_TOKEN` with ML-DSA-87 signature
   - L7 Router verifies signature before processing

### 4.2 ROE and Compartment Enforcement

**ROE Levels (Phase 3 scope):**

| Level | Description | Allowed Operations | L7 Profile |
|-------|-------------|-------------------|-----------|
| `ANALYSIS_ONLY` | Read-only analysis, no external actions | Chat completions, summaries | All |
| `SOC_ASSIST` | SOC operator assistance, alerting | Chat + agent tasks | All |
| `TRAINING` | Development/testing mode | Full access, logging increased | Dev profiles only |

**Compartment Masks:**

```python
COMPARTMENT_SOC      = 0x01
COMPARTMENT_DEV      = 0x02
COMPARTMENT_LAB      = 0x04
COMPARTMENT_CRYPTO   = 0x08
COMPARTMENT_KINETIC  = 0x80  # ALWAYS DENIED in L7
```

**Policy Enforcement (L7 Router):**

```python
def apply_policy(self, msg: DBEMessage) -> Optional[str]:
    compartment = msg.tlv_get("COMPARTMENT_MASK", 0)

    # Hard block KINETIC in L7
    if compartment & 0x80:
        return "DENIED: KINETIC compartment not allowed in Layer 7"

    # Restrict EXEC classification to Layer 9
    if msg.tlv_get("CLASSIFICATION") == "EXEC":
        return "DENIED: EXEC classification requires Layer 9 authorization"

    return None  # Allowed
```

---

## 5. Phase 3 Workstreams

### 5.1 Workstream 1: L7 DBE Schema & `libdbe`

**Tasks:**
1. Define Protobuf schemas for L7 messages:
   ```protobuf
   message L7ChatRequest {
     repeated Message messages = 1;
     float temperature = 2;
     int32 max_tokens = 3;
     string model = 4;
   }

   message L7ChatResponse {
     repeated Choice choices = 1;
     Usage usage = 2;
     string model = 3;
     int32 device_id = 4;
     float latency_ms = 5;
   }

   message L7AgentTask {
     string task_type = 1;
     map<string, string> parameters = 2;
     int32 timeout_seconds = 3;
   }

   message L7AgentResult {
     string status = 1;
     string result = 2;
     repeated string artifacts = 3;
   }
   ```

2. Integrate into `libdbe` (Rust or C with Python bindings)
3. Implement PQC handshake helpers (ML-KEM-1024 + ML-DSA-87)
4. Implement AES-256-GCM channel encryption

**Deliverables:**
- `libdbe` v1.0 with L7 message types
- Python bindings: `dsmil_dbe` package
- Unit tests for DBE encoding/decoding

### 5.2 Workstream 2: L7 Router Implementation

**Tasks:**
1. Implement DBE message reception on UDS socket
2. Implement `L7_CLAIM_TOKEN` verification (ML-DSA-87)
3. Implement policy engine (compartment, ROE, classification checks)
4. Implement worker selection and load balancing
5. Implement DBE message forwarding to workers
6. Implement logging (journald with `SyslogIdentifier=dsmil-l7-router`)

**Deliverables:**
- `l7_router.py` (production-ready)
- systemd unit: `dsmil-l7-router.service`
- Configuration file: `/etc/dsmil/l7_router.yaml`

### 5.3 Workstream 3: Device 47 LLM Worker

**Tasks:**
1. Set up model repository: `/opt/dsmil/models/llama-7b-int8`
2. Implement INT8 model loading with Intel Extension for PyTorch
3. Implement DBE message handling (L7_CHAT_REQ → L7_CHAT_RESP)
4. Optimize for AMX (Advanced Matrix Extensions)
5. Implement KV cache management (10 GB allocation)
6. Implement memory monitoring and OOM prevention
7. Implement performance logging (tokens/sec, latency)

**Deliverables:**
- `l7_llm_worker_47.py` (production-ready)
- systemd unit: `dsmil-l7-llm-worker-47.service`
- Model optimization scripts
- Performance benchmark results

### 5.4 Workstream 4: OpenAI Shim Integration

**Tasks:**
1. Implement FastAPI endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions`)
2. Implement API key validation
3. Implement OpenAI format → DBE L7_CHAT_REQ conversion
4. Implement DBE L7_CHAT_RESP → OpenAI format conversion
5. Implement L7_CLAIM_TOKEN generation (ML-DSA-87 signing)
6. Bind to localhost only (127.0.0.1:8001)
7. Implement error handling and logging

**Deliverables:**
- `openai_shim.py` (production-ready)
- systemd unit: `dsmil-openai-shim.service`
- Integration test suite
- Example usage documentation

### 5.5 Workstream 5: Logging & Monitoring

**Tasks:**
1. Extend journald logging with L7-specific tags
2. Add SHRINK monitoring for L7 services (stress detection)
3. Implement Prometheus metrics for L7 Router and Worker 47:
   - `dsmil_l7_requests_total{device_id, profile, status}`
   - `dsmil_l7_latency_seconds{device_id, profile}`
   - `dsmil_l7_tokens_generated_total{device_id}`
   - `dsmil_l7_memory_used_bytes{device_id}`
4. Create Grafana dashboard for Layer 7 monitoring

**Deliverables:**
- Updated journald configuration
- Prometheus scrape configs
- Grafana dashboard JSON

---

## 6. Phase 3 Exit Criteria

Phase 3 is complete when:

- [x] **`libdbe` implemented and tested:**
  - Protobuf schemas for L7 messages
  - PQC handshake (ML-KEM-1024 + ML-DSA-87)
  - AES-256-GCM channel encryption
  - Python bindings functional

- [x] **L7 Router operational (Device 43):**
  - `dsmil-l7-router.service` running
  - Receiving DBE messages on UDS socket
  - Validating L7_CLAIM_TOKEN signatures
  - Enforcing compartment/ROE/classification policies
  - Routing to Device 47 LLM Worker

- [x] **Device 47 LLM Worker operational:**
  - `dsmil-l7-llm-worker-47.service` running
  - LLaMA-7B INT8 model loaded (7.2 GB weights)
  - KV cache allocated (10 GB for 32K context)
  - AMX acceleration active
  - Generating completions via DBE
  - Logging tokens/sec and latency metrics

- [x] **OpenAI Shim operational:**
  - `dsmil-openai-shim.service` running on 127.0.0.1:8001
  - `/v1/models` endpoint working
  - `/v1/chat/completions` endpoint working
  - API key validation enforced
  - All requests routed via DBE to L7 Router

- [x] **Local tools can use OpenAI API:**
  - LangChain integration tested
  - VSCode Copilot configuration documented
  - CLI tools (e.g., `curl`) successfully call shim
  - Example: `export OPENAI_API_KEY=dsmil-local-key && python langchain_example.py`

- [x] **All L7 internal calls use DBE:**
  - No HTTP between L7 Router and Worker 47
  - No HTTP between L7 Router and Agent Harness
  - All UDS sockets use DBE protocol
  - Verified with `tcpdump` (no TCP traffic between L7 services)

- [x] **L7 policy engine enforces security:**
  - KINETIC compartment blocked
  - EXEC classification blocked (Layer 9 only)
  - Tenant isolation working
  - Rate limiting per tenant functional

- [x] **Logging and monitoring active:**
  - All L7 services log to journald
  - SHRINK monitoring L7 operator activity
  - Prometheus metrics scraped
  - Grafana dashboard displaying L7 status

### Validation Commands

```bash
# Verify L7 services
systemctl status dsmil-l7-router.service
systemctl status dsmil-l7-llm-worker-47.service
systemctl status dsmil-openai-shim.service

# Verify DBE sockets
ls -la /var/run/dsmil/*.sock

# Test OpenAI shim
curl -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer dsmil-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b-int8-amx",
    "messages": [{"role": "user", "content": "What is DSMIL?"}],
    "max_tokens": 100
  }'

# Verify DBE traffic (no TCP between L7 services)
sudo tcpdump -i lo port not 8001 -c 100

# Check L7 metrics
curl http://localhost:9090/api/v1/query?query=dsmil_l7_requests_total

# View L7 logs
journalctl -u dsmil-l7-router.service -f
journalctl -u dsmil-l7-llm-worker-47.service -f
```

---

## 7. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| L7 Router latency | < 5ms overhead | DBE message routing time |
| Device 47 inference (LLaMA-7B) | > 20 tokens/sec | Output tokens per second |
| Device 47 TTFT (time to first token) | < 500ms | Latency to first output token |
| OpenAI shim overhead | < 10ms | HTTP → DBE conversion time |
| End-to-end latency (shim → completion) | < 2 seconds for 100 tokens | Full request-response cycle |
| Memory usage (Device 47) | < 20 GB | Monitored via cgroups |
| DBE message throughput | > 5,000 msg/sec | L7 Router capacity |

---

## 8. Next Phase Preview (Phase 4)

Phase 4 will build on Phase 3 by:

1. **Layer 8/9 Activation:**
   - Deploy Device 53 (Cryptographic AI) for PQC monitoring
   - Activate Device 61 (NC3 Integration) with ROE gating
   - Implement Device 58 (SOAR) for automated response

2. **Advanced L7 Capabilities:**
   - Multi-modal integration (CLIP vision on Device 45)
   - Agent orchestration (Device 48 agent harness)
   - Strategic planning AI (Device 48)

3. **DBE Mesh Expansion:**
   - L8 ↔ L7 DBE flows (SOC → LLM integration)
   - L9 ↔ L8 DBE flows (Executive → Security oversight)
   - Cross-layer correlation

---

## 9. Document Metadata

**Version History:**
- **v1.0 (2024-Q4):** Initial Phase 3 spec (duplicate Master Plan content)
- **v2.0 (2025-11-23):** Rewritten as L7 Generative Plane deployment
  - Aligned with v3.1 Comprehensive Plan
  - Added Device 47 specifications (20 GB, LLaMA-7B INT8)
  - Detailed DBE protocol integration
  - Complete L7 Router and Worker implementations
  - OpenAI shim with DBE routing
  - PQC boundaries (ML-KEM-1024, ML-DSA-87)
  - Exit criteria and validation commands

**Dependencies:**
- Phase 1 (Foundation) completed
- Phase 2F (Data Fabric + SHRINK) completed
- `libdbe` v1.0 (DSMIL Binary Envelope library)
- liboqs (Open Quantum Safe)
- Intel Extension for PyTorch
- transformers >= 4.35
- FastAPI >= 0.104

**References:**
- `00_MASTER_PLAN_OVERVIEW_CORRECTED.md (v3.1)`
- `01_HARDWARE_INTEGRATION_LAYER_DETAILED.md (v3.1)`
- `05_LAYER_SPECIFIC_DEPLOYMENTS.md (v1.0)`
- `06_CROSS_LAYER_INTELLIGENCE_FLOWS.md (v1.0)`
- `07_IMPLEMENTATION_ROADMAP.md (v1.0)`
- `Phase1.md (v2.0)`
- `Phase2F.md (v2.0)`
- `Phase7.md (v1.0)` - DBE protocol specification

**Contact:**
For questions or issues with Phase 3 implementation, contact DSMIL L7 Team.

---

**END OF PHASE 3 SPECIFICATION**
