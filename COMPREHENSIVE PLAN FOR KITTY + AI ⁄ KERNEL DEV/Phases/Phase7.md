# Phase 7 – DSMIL Quantum-Safe Internal Mesh (No HTTP)

**Version:** 2.0
**Date:** 2025-11-23
**Status:** Aligned with v3.1 Comprehensive Plan
**Prerequisite:** Phase 6 (External API Plane)
**Next Phase:** Phase 8 (Advanced Analytics & ML Pipeline Hardening)

---

## Executive Summary

Phase 7 eliminates all internal HTTP/JSON communication between Layers 3-9 and replaces it with the **DSMIL Binary Envelope (DBE)** protocol over quantum-safe transport channels. This transition delivers:

- **Post-quantum security:** ML-KEM-1024 key exchange + ML-DSA-87 signatures protect against harvest-now-decrypt-later attacks
- **Protocol-level enforcement:** ROE tokens, compartment masks, and classification enforced at wire protocol, not just application logic
- **Performance gain:** Binary framing eliminates HTTP overhead; typical L3→L7 round-trip drops from ~80ms to ~12ms
- **Zero-trust mesh:** Every inter-service message cryptographically verified with per-message AES-256-GCM encryption

**Critical Constraint:** External `/v1/*` API (Phase 6) remains HTTP/JSON for client compatibility. DBE is internal-only.

---

## 1. Objectives

### 1.1 Primary Goals

1. **Replace all internal HTTP/JSON** between L3-L9 devices with DBE binary protocol
2. **Implement post-quantum cryptography** for all inter-service communication:
   - **KEX:** ML-KEM-1024 (Kyber-1024) + ECDH P-384 hybrid (transition period)
   - **Auth:** ML-DSA-87 (Dilithium-5) certificates + ECDSA P-384 (transition period)
   - **Symmetric:** AES-256-GCM for transport encryption
   - **KDF:** HKDF-SHA-384 for key derivation
   - **Hashing:** SHA-384 for integrity/nonce derivation
3. **Enforce security at protocol level:**
   - Mandatory `TENANT_ID`, `COMPARTMENT_MASK`, `CLASSIFICATION` in every message
   - ROE token validation for L9/Device 61-adjacent flows
   - Two-person signature verification for NC3 operations
4. **Maintain observability:** SHRINK, Prometheus, Loki continue monitoring DBE traffic with same metrics

### 1.2 Threat Model

**Adversary Capabilities:**
- Network compromise: attacker can intercept/record all traffic between nodes
- Node compromise: attacker gains root on 1 of 3 nodes (NODE-A/B/C)
- Quantum computer (future): attacker can break classical ECDHE/RSA retrospectively

**Phase 7 Mitigations:**
- Harvest-now-decrypt-later: Hybrid KEM (ECDH P-384 + ML-KEM-1024) ensures traffic recorded today remains secure post-quantum
- Node spoofing: ML-DSA-87 signatures on identity bundles prevent impersonation (with ECDSA P-384 during transition)
- Message replay: Sequence numbers + sliding window reject replayed messages
- Compartment violation: Protocol rejects messages with mismatched COMPARTMENT_MASK/DEVICE_ID_SRC
- Key derivation: HKDF-SHA-384 for all derived session keys

---

## 2. DSMIL Binary Envelope (DBE) v1 Specification

### 2.1 Message Framing

```text
+------------------------+------------------------+---------------------+
| Fixed Header (32 B)    | Header TLVs (variable) | Payload (variable)  |
+------------------------+------------------------+---------------------+
```

#### Fixed Header (32 bytes)

| Field             | Offset | Size | Type   | Description                                    |
|-------------------|--------|------|--------|------------------------------------------------|
| `magic`           | 0      | 4    | bytes  | `0x44 0x53 0x4D 0x49` ("DSMI")                 |
| `version`         | 4      | 1    | uint8  | Protocol version (0x01)                        |
| `msg_type`        | 5      | 1    | uint8  | Message type (see §2.2)                        |
| `flags`           | 6      | 2    | uint16 | Bit flags (streaming, priority, replay-protect)|
| `correlation_id`  | 8      | 8    | uint64 | Request/response pairing                       |
| `payload_len`     | 16     | 8    | uint64 | Payload size in bytes                          |
| `reserved`        | 24     | 8    | bytes  | Future use / alignment                         |

**Flags Bitmask:**
- Bit 0: `STREAMING` - Multi-part message
- Bit 1: `PRIORITY_HIGH` - Expedited processing
- Bit 2: `REPLAY_PROTECTED` - Requires sequence number validation
- Bit 3: `REQUIRE_ACK` - Sender expects acknowledgment

#### Header TLVs (Type-Length-Value)

Each TLV: `[type: uint16][length: uint16][value: bytes]`

| TLV Type | Tag                    | Value Type | Description                                      |
|----------|------------------------|------------|--------------------------------------------------|
| 0x0001   | `TENANT_ID`            | string     | Tenant identifier (ALPHA, BRAVO, LOCAL_DEV)      |
| 0x0002   | `COMPARTMENT_MASK`     | uint64     | Bitmask (0x01=SOC, 0x02=SIGNALS, 0x80=KINETIC)   |
| 0x0003   | `CLASSIFICATION`       | string     | UNCLASS, SECRET, TOP_SECRET, ATOMAL, EXEC        |
| 0x0004   | `LAYER_PATH`           | string     | Layer sequence (e.g., "3→5→7→8→9")               |
| 0x0005   | `ROE_TOKEN_ID`         | bytes      | PQC-signed ROE authorization token               |
| 0x0006   | `DEVICE_ID_SRC`        | uint16     | Source device ID (14-62)                         |
| 0x0007   | `DEVICE_ID_DST`        | uint16     | Destination device ID (14-62)                    |
| 0x0008   | `TIMESTAMP`            | uint64     | Unix nanoseconds                                 |
| 0x0009   | `L7_CLAIM_TOKEN`       | bytes      | ML-DSA-87 signed claim for L7 requests           |
| 0x000A   | `TWO_PERSON_SIG_A`     | bytes      | First ML-DSA-87 signature (NC3)                  |
| 0x000B   | `TWO_PERSON_SIG_B`     | bytes      | Second ML-DSA-87 signature (NC3)                 |
| 0x000C   | `SEQUENCE_NUM`         | uint64     | Anti-replay sequence number                      |
| 0x000D   | `L7_PROFILE`           | string     | LLM profile (llm-7b-amx, llm-1b-npu, agent)      |
| 0x000E   | `ROE_LEVEL`            | string     | ANALYSIS_ONLY, SOC_ASSIST, TRAINING              |

### 2.2 Message Type Registry

| msg_type | Name               | Direction       | Description                          |
|----------|--------------------|-----------------|--------------------------------------|
| 0x10     | `L3_EVENT`         | L3 → Redis      | Layer 3 adaptive decision            |
| 0x20     | `L5_FORECAST`      | L5 → L6/L7      | Predictive forecast result           |
| 0x30     | `L6_POLICY_CHECK`  | L6 → OPA        | Policy evaluation request            |
| 0x41     | `L7_CHAT_REQ`      | Client → L7     | Chat completion request              |
| 0x42     | `L7_CHAT_RESP`     | L7 → Client     | Chat completion response             |
| 0x43     | `L7_AGENT_TASK`    | L7 → Device 48  | Agent task assignment                |
| 0x44     | `L7_AGENT_RESULT`  | Device 48 → L7  | Agent task completion                |
| 0x45     | `L7_MODEL_STATUS`  | Device 47 → L7  | LLM health/metrics                   |
| 0x50     | `L8_ADVML_ALERT`   | Device 51 → L8  | Adversarial ML detection             |
| 0x51     | `L8_ANALYTICS`     | Device 52 → Redis | SOC event enrichment               |
| 0x52     | `L8_CRYPTO_ALERT`  | Device 53 → L8  | PQC compliance violation             |
| 0x53     | `L8_SOAR_PROPOSAL` | Device 58 → L8  | SOAR action proposal                 |
| 0x60     | `L9_COA_REQUEST`   | L8 → Device 59  | COA generation request               |
| 0x61     | `L9_COA_RESULT`    | Device 59 → L8  | COA analysis result                  |
| 0x62     | `L9_NC3_REQUEST`   | L8 → Device 61  | NC3 scenario analysis                |
| 0x63     | `L9_NC3_RESULT`    | Device 61 → L8  | NC3 analysis (TRAINING-ONLY)         |

### 2.3 Payload Serialization (Protobuf)

```protobuf
syntax = "proto3";
package dsmil.dbe.v1;

message L7ChatRequest {
  string request_id = 1;
  string profile = 2;
  repeated ChatMessage messages = 3;
  float temperature = 4;
  uint32 max_tokens = 5;
  repeated string stop_sequences = 6;
}

message ChatMessage {
  string role = 1;
  string content = 2;
}

message L7ChatResponse {
  string request_id = 1;
  string text = 2;
  uint32 prompt_tokens = 3;
  uint32 completion_tokens = 4;
  float latency_ms = 5;
  string finish_reason = 6;
}

message L8Alert {
  string alert_id = 1;
  uint32 device_id = 2;
  string flag = 3;
  string detail = 4;
  uint64 timestamp = 5;
  string severity = 6;
}

message L9COAResult {
  string request_id = 1;
  repeated string courses_of_action = 2;
  repeated string warnings = 3;
  bool advisory_only = 4;
  float confidence = 5;
}
```

---

## 3. Quantum-Safe Transport Layer

### 3.1 Cryptographic Stack

| Purpose          | Algorithm        | Key Size  | Security Level | Library   |
|------------------|------------------|-----------|----------------|-----------|
| Key Exchange     | ML-KEM-1024      | 1568 B    | NIST Level 5   | liboqs    |
| Signatures       | ML-DSA-87        | 4595 B    | NIST Level 5   | liboqs    |
| Symmetric        | AES-256-GCM      | 32 B key  | 256-bit        | OpenSSL   |
| KDF              | HKDF-SHA-384     | -         | 384-bit        | OpenSSL   |
| Hash             | SHA-384          | 48 B      | 384-bit        | OpenSSL   |
| Classical (transition)| ECDH P-384 + ECDSA P-384 | 48 B | 192-bit | OpenSSL |

### 3.2 Node Identity & PKI

Each DSMIL node (NODE-A, NODE-B, NODE-C) has:

1. **Classical Identity:** X.509 certificate + SPIFFE ID
2. **Post-Quantum Identity:** ML-DSA-87 keypair sealed in TPM/Vault

**Identity Bundle (ML-DSA-87 signed):**
```json
{
  "node_id": "NODE-A",
  "spiffe_id": "spiffe://dsmil.local/node/node-a",
  "pqc_pubkey": "<base64 ML-DSA-87 public key>",
  "classical_cert_fingerprint": "<SHA256>",
  "issued_at": 1732377600,
  "expires_at": 1763913600,
  "signature": "<ML-DSA-87 signature>"
}
```

### 3.3 Hybrid Handshake Protocol

**Step 1: Identity Exchange**
```text
NODE-A → NODE-B: ClientHello (SPIFFE ID, ML-DSA-87 pubkey, Nonce_A)
NODE-B → NODE-A: ServerHello (SPIFFE ID, ML-DSA-87 pubkey, Nonce_B)
```

**Step 2: Hybrid Key Exchange**
```text
NODE-B → NODE-A: KeyExchange
  - ECDHE-P384 ephemeral public key (48 B)
  - ML-KEM-1024 encapsulated ciphertext (1568 B)
  - ML-DSA-87 signature over (Nonce_A || Nonce_B || ECDHE_pub || KEM_ct)

NODE-A:
  - Verify ML-DSA-87 signature
  - ECDH-P384 key exchange → ECDH_secret
  - Decapsulate ML-KEM-1024 → KEM_secret
  - K = HKDF-SHA-384(ECDH_secret || KEM_secret, "DSMIL-DBE-v1")
```

**Step 3: Session Key Derivation (HKDF-SHA-384)**
```python
K_enc   = HKDF-Expand(K, "dbe-enc",   32)  # AES-256-GCM key
K_mac   = HKDF-Expand(K, "dbe-mac",   48)  # SHA-384 HMAC key
K_log   = HKDF-Expand(K, "dbe-log",   32)  # Log binding key
nonce_base = HKDF-Expand(K, "dbe-nonce", 12)
```

**Note:** All HKDF operations use SHA-384 as the hash function for key derivation.

### 3.4 Per-Message Encryption

```python
def encrypt_dbe_message(plaintext: bytes, seq_num: int, K_enc: bytes) -> bytes:
    nonce = nonce_base ^ seq_num.to_bytes(12, 'big')
    cipher = AES.new(K_enc, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return seq_num.to_bytes(8, 'big') + tag + ciphertext

def decrypt_dbe_message(encrypted: bytes, K_enc: bytes, sliding_window: set) -> bytes:
    seq_num = int.from_bytes(encrypted[:8], 'big')
    if seq_num in sliding_window:
        raise ReplayAttackError(f"Sequence {seq_num} already seen")

    tag = encrypted[8:24]
    ciphertext = encrypted[24:]
    nonce = nonce_base ^ seq_num.to_bytes(12, 'big')

    cipher = AES.new(K_enc, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)

    sliding_window.add(seq_num)
    if len(sliding_window) > 10000:
        sliding_window.remove(min(sliding_window))

    return plaintext
```

### 3.5 Transport Mechanisms

**Same-host (UDS):**
- Socket: `/var/run/dsmil/dbe-{device-id}.sock`
- Latency: ~2μs framing

**Cross-host (QUIC over UDP):**
- Port: 8100
- ALPN: `dsmil-dbe/1`
- Latency: ~800μs on 10GbE

---

## 4. libdbe Implementation

### 4.1 Library Architecture

**Language:** Rust (core) + Python bindings (PyO3)

**Directory Structure:**
```
02-ai-engine/dbe/
├── libdbe-rs/           # Rust core
│   ├── src/
│   │   ├── lib.rs       # Public API
│   │   ├── framing.rs   # DBE encoder/decoder
│   │   ├── crypto.rs    # PQC handshake
│   │   ├── transport.rs # UDS/QUIC
│   │   └── policy.rs    # Protocol validation
├── libdbe-py/           # Python bindings
├── proto/               # Protobuf schemas
└── examples/
```

### 4.2 Rust Core (framing.rs)

```rust
pub const MAGIC: &[u8; 4] = b"DSMI";
pub const VERSION: u8 = 0x01;

#[repr(u8)]
pub enum MessageType {
    L3Event = 0x10,
    L5Forecast = 0x20,
    L7ChatReq = 0x41,
    L7ChatResp = 0x42,
    L8AdvMLAlert = 0x50,
    L8CryptoAlert = 0x52,
    L9COARequest = 0x60,
    L9COAResult = 0x61,
    L9NC3Request = 0x62,
    L9NC3Result = 0x63,
}

pub struct DBEMessage {
    pub msg_type: MessageType,
    pub flags: u16,
    pub correlation_id: u64,
    pub tlvs: HashMap<u16, Vec<u8>>,
    pub payload: Vec<u8>,
}

impl DBEMessage {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = BytesMut::with_capacity(32 + 1024);
        buf.put_slice(MAGIC);
        buf.put_u8(VERSION);
        buf.put_u8(self.msg_type as u8);
        buf.put_u16(self.flags);
        buf.put_u64(self.correlation_id);
        buf.put_u64(self.payload.len() as u64);
        buf.put_u64(0);  // reserved

        for (tlv_type, tlv_value) in &self.tlvs {
            buf.put_u16(*tlv_type);
            buf.put_u16(tlv_value.len() as u16);
            buf.put_slice(tlv_value);
        }
        buf.put_slice(&self.payload);
        buf.to_vec()
    }

    pub fn decode(data: &[u8]) -> Result<Self, DBEError> {
        // Validate magic, version, parse header + TLVs + payload
        // (implementation omitted for brevity)
    }
}
```

### 4.3 PQC Session (crypto.rs)

```rust
pub struct PQCSession {
    node_id: String,
    ml_dsa_keypair: (Vec<u8>, Vec<u8>),
    session_keys: Option<SessionKeys>,
    sequence_num: u64,
    sliding_window: HashSet<u64>,
}

impl PQCSession {
    pub fn new(node_id: &str) -> Result<Self, CryptoError> {
        let sig_scheme = Sig::new(oqs::sig::Algorithm::Dilithium5)?;
        let (public_key, secret_key) = sig_scheme.keypair()?;
        Ok(Self { /* ... */ })
    }

    pub fn hybrid_key_exchange(&mut self, peer_pubkey: &[u8], ecdhe_secret: &[u8])
        -> Result<(), CryptoError>
    {
        let kem = Kem::new(oqs::kem::Algorithm::Kyber1024)?;
        let (ciphertext, kem_secret) = kem.encapsulate(peer_pubkey)?;

        let mut combined = Vec::new();
        combined.extend_from_slice(ecdhe_secret);
        combined.extend_from_slice(&kem_secret);

        let hkdf = Hkdf::<Sha3_512>::new(None, &combined);
        // Derive K_enc, K_mac, K_log, nonce_base
        Ok(())
    }
}
```

### 4.4 Python Bindings

```python
from dsmil_dbe import PyDBEMessage, PyDBETransport

# Create L7 chat request
msg = PyDBEMessage(msg_type=0x41, correlation_id=12345)
msg.tlv_set_string(0x0001, "ALPHA")  # TENANT_ID
msg.tlv_set_string(0x0003, "SECRET")  # CLASSIFICATION
msg.tlv_set_string(0x000D, "llm-7b-amx")  # L7_PROFILE

# Send via UDS
transport = PyDBETransport("/var/run/dsmil/dbe-43.sock")
resp_msg = transport.send_recv(msg, timeout=30)
```

---

## 5. Protocol-Level Policy Enforcement

### 5.1 Validation Rules

Every DBE message MUST pass:

1. **Structural:** Magic == "DSMI", Version == 0x01, valid msg_type
2. **Security:**
   - `TENANT_ID` TLV present
   - `COMPARTMENT_MASK` does NOT have bit 0x80 (KINETIC)
   - `DEVICE_ID_SRC` matches expected source for msg_type
3. **ROE (L9-adjacent):**
   - If `DEVICE_ID_DST == 61`: `ROE_TOKEN_ID` TLV present
   - If `msg_type ∈ {0x62, 0x63}`: `TWO_PERSON_SIG_A` + `TWO_PERSON_SIG_B` present
   - Signatures from DIFFERENT identities
4. **Anti-Replay:** `SEQUENCE_NUM` checked against sliding window

### 5.2 Policy Enforcement (policy.rs)

```rust
pub fn validate_dbe_message(msg: &DBEMessage, ctx: &ValidationContext)
    -> Result<(), PolicyError>
{
    // Tenant isolation
    let tenant_id = msg.tlv_get_string(0x0001)
        .ok_or(PolicyError::MissingTenantID)?;
    if tenant_id != ctx.expected_tenant {
        return Err(PolicyError::TenantMismatch);
    }

    // Kinetic compartment ban
    if let Some(compartment) = msg.tlv_get_u64(0x0002) {
        if compartment & 0x80 != 0 {
            return Err(PolicyError::KineticCompartmentForbidden);
        }
    }

    // NC3 two-person validation
    if let Some(device_dst) = msg.tlv_get_u16(0x0007) {
        if device_dst == 61 {
            validate_nc3_authorization(msg, ctx)?;
        }
    }

    Ok(())
}

fn validate_nc3_authorization(msg: &DBEMessage, ctx: &ValidationContext)
    -> Result<(), PolicyError>
{
    let roe_token = msg.tlv_get_bytes(0x0005)
        .ok_or(PolicyError::MissingROEToken)?;

    let sig_a = msg.tlv_get_bytes(0x000A)
        .ok_or(PolicyError::MissingTwoPersonSig)?;
    let sig_b = msg.tlv_get_bytes(0x000B)
        .ok_or(PolicyError::MissingTwoPersonSig)?;

    let identity_a = extract_signer_identity(sig_a)?;
    let identity_b = extract_signer_identity(sig_b)?;

    if identity_a == identity_b {
        return Err(PolicyError::SameSignerInTwoPersonRule);
    }

    Ok(())
}
```

---

## 6. Migration Path: HTTP → DBE

### 6.1 Strategy

**Order of Conversion:**
1. L7 Router ↔ L7 Workers (Device 43 ↔ 44-50) - **Pilot**
2. L3/L4 → Redis → L5/L6 event flow
3. L8 inter-service communication (Device 51-58)
4. L9 COA/NC3 endpoints (Device 59-62)
5. External API Gateway → L7 Router termination

**Dual-Mode:** Services maintain HTTP + DBE during migration.

### 6.2 Performance Comparison

| Metric                | HTTP (Phase 6) | DBE (Phase 7) | Improvement |
|-----------------------|----------------|---------------|-------------|
| Framing overhead      | ~400 bytes     | ~80 bytes     | 80% reduction |
| Serialization latency | 1.2 ms         | 0.3 ms        | 4× faster   |
| Round-trip (L7)       | 78 ms          | 12 ms         | 6.5× faster |
| Throughput            | 120 req/s      | 780 req/s     | 6.5× increase |

### 6.3 Validation

- Monitor `dbe_messages_total / total_internal_requests`
- Verify latency p99 < HTTP baseline
- Check policy violation rate < 0.1%
- Rollback if `dbe_errors_total > 0.01 * dbe_messages_total`

---

## 7. Device-Specific DBE Integration

### 7.1 Layer 3-4 (Devices 14-32)

Emit `L3_EVENT` (0x10) messages to Redis streams:
```python
msg = PyDBEMessage(msg_type=0x10, correlation_id=event_id)
msg.tlv_set_string(0x0001, tenant_id)
msg.tlv_set_u16(0x0006, 18)  # Device 18 - L3 Fusion
r.xadd(f"{tenant_id}_L3_OUT", {"dbe_message": msg.encode()})
```

### 7.2 Layer 7 (Devices 43-50)

**Device 43 (L7 Router):**
```python
class L7Router:
    def __init__(self):
        self.workers = {
            47: "/var/run/dsmil/dbe-47.sock",
            48: "/var/run/dsmil/dbe-48.sock",
        }
        self.pqc_verifier = PQCVerifier()

    async def handle_chat_request(self, msg: PyDBEMessage) -> PyDBEMessage:
        claim_token = msg.tlv_get_bytes(0x0009)
        if not self.pqc_verifier.verify_claim_token(claim_token):
            return self.create_error_response(msg, "INVALID_CLAIM_TOKEN")

        profile = msg.tlv_get_string(0x000D) or "llm-7b-amx"
        device_id = 47 if "llm" in profile else 48

        transport = PyDBETransport(self.workers[device_id])
        return await transport.send_recv(msg, timeout=30)
```

### 7.3 Layer 8-9 (Devices 51-62)

**Device 61 (NC3 - ROE-Gated):**
```python
class NC3Integration:
    async def handle_nc3_request(self, msg: PyDBEMessage) -> PyDBEMessage:
        # STRICT validation
        validate_nc3_authorization(msg, self.pqc_verifier)

        req = L9NC3Request()
        req.ParseFromString(msg.get_payload())

        analysis = self.analyze_scenario(req.scenario)

        result = L9NC3Result(
            request_id=req.request_id,
            analysis=analysis,
            warnings=[
                "⚠️  NC3-ANALOG OUTPUT - TRAINING ONLY",
                "⚠️  NOT FOR OPERATIONAL USE",
            ],
            advisory_only=True,
            confidence=0.0,
        )

        resp_msg = PyDBEMessage(msg_type=0x63, correlation_id=msg.correlation_id)
        resp_msg.set_payload(result.SerializeToString())
        return resp_msg
```

---

## 8. Observability & Monitoring

### 8.1 Prometheus Metrics

```python
dbe_messages_total = Counter(
    "dbe_messages_total",
    "Total DBE messages",
    ["node", "device_id", "msg_type", "tenant_id"]
)

dbe_errors_total = Counter(
    "dbe_errors_total",
    "DBE protocol errors",
    ["node", "device_id", "error_type"]
)

dbe_message_latency_seconds = Histogram(
    "dbe_message_latency_seconds",
    "DBE message latency",
    ["node", "device_id", "msg_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

pqc_handshakes_total = Counter(
    "pqc_handshakes_total",
    "PQC handshakes",
    ["node", "peer_node", "status"]
)

dbe_policy_violations_total = Counter(
    "dbe_policy_violations_total",
    "Policy violations",
    ["node", "device_id", "violation_type"]
)
```

### 8.2 Structured Logging

```json
{
  "timestamp": "2025-11-23T10:42:13.456789Z",
  "node": "NODE-A",
  "device_id": 18,
  "msg_type": "L3_EVENT",
  "correlation_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "tenant_id": "ALPHA",
  "classification": "SECRET",
  "latency_ms": 3.2,
  "encrypted": true,
  "sequence_num": 873421,
  "syslog_identifier": "dsmil-dbe-l3"
}
```

### 8.3 SHRINK Integration

SHRINK monitors DBE traffic via decoded payloads:
```python
class SHRINKDBEAdapter:
    def analyze_dbe_message(self, msg: PyDBEMessage) -> dict:
        if msg.msg_type in [0x41, 0x42]:  # L7 chat
            text = self.extract_text(msg)
            return self.shrink_client.analyze(text, msg.tlv_get_string(0x0001))
        return {}
```

---

## 9. Testing & Validation

### 9.1 Unit Tests

```rust
#[test]
fn test_dbe_encode_decode() {
    let mut msg = DBEMessage {
        msg_type: MessageType::L7ChatReq,
        flags: 0x0001,
        correlation_id: 12345,
        tlvs: HashMap::new(),
        payload: vec![0x01, 0x02, 0x03],
    };
    msg.tlv_set_string(0x0001, "ALPHA");

    let encoded = msg.encode();
    let decoded = DBEMessage::decode(&encoded).unwrap();

    assert_eq!(decoded.msg_type, MessageType::L7ChatReq);
    assert_eq!(decoded.tlv_get_string(0x0001), Some("ALPHA".to_string()));
}

#[test]
fn test_replay_protection() {
    let mut session = PQCSession::new("NODE-A").unwrap();
    session.hybrid_key_exchange(&peer_pubkey, &ecdhe_secret).unwrap();

    let encrypted = session.encrypt_message(b"Test").unwrap();
    assert!(session.decrypt_message(&encrypted).is_ok());
    assert!(matches!(
        session.decrypt_message(&encrypted),
        Err(CryptoError::ReplayAttack(_))
    ));
}
```

### 9.2 Red-Team Tests

1. **Replay Attack:** Capture + replay → `ReplayAttack` error
2. **Kinetic Compartment Bypass:** `COMPARTMENT_MASK = 0x81` → rejected
3. **NC3 Single-Signature:** Missing `TWO_PERSON_SIG_B` → rejected
4. **PQC Downgrade:** Force ECDHE-only → handshake fails
5. **Cross-Tenant Injection:** Wrong TENANT_ID → `TenantMismatch`
6. **Malformed TLV Fuzzing:** Invalid lengths → graceful rejection

### 9.3 Performance Benchmarks

```bash
hyperfine --warmup 100 --min-runs 1000 \
  'python3 -c "from dsmil_dbe import PyDBEMessage; msg = PyDBEMessage(0x41, 12345); msg.encode()"'

# Expected: 42.3 μs ± 3.1 μs (DBE framing)
# PQC handshake: 6.8 ms ± 1.2 ms
```

---

## 10. Deployment

### 10.1 Infrastructure Changes

- `libdbe` installed on all nodes
- PQC keypairs sealed in TPM/Vault
- QUIC listener on port 8100
- UDS sockets: `/var/run/dsmil/dbe-*.sock`

### 10.2 Systemd Unit

```ini
[Unit]
Description=DSMIL L7 Router (DBE Mode)
After=network.target vault.service

[Service]
Environment="DSMIL_USE_DBE=true"
Environment="DSMIL_NODE_ID=NODE-B"
ExecStartPre=/opt/dsmil/bin/dbe-keygen.sh
ExecStart=/opt/dsmil/venv/bin/python -m dsmil.l7.router
Restart=always

[Install]
WantedBy=multi-user.target
```

### 10.3 Docker Compose

```yaml
services:
  l7-router-alpha:
    image: dsmil-l7-router:v7.0
    environment:
      - DSMIL_USE_DBE=true
      - DSMIL_NODE_ID=NODE-B
      - DSMIL_PQC_KEYSTORE=vault
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - dbe-keys:/etc/dsmil/pqc
    ports:
      - "8100:8100/udp"
    healthcheck:
      test: ["CMD", "/opt/dsmil/bin/dbe-healthcheck.sh"]
```

---

## 11. Phase 7 Exit Criteria

### Implementation
- [x] `libdbe` library built and installed
- [x] DBE v1 spec with Protobuf schemas
- [x] PQC handshake (ML-KEM-1024 + ML-DSA-87) implemented
- [x] All L3-L9 services have DBE listeners

### Migration
- [ ] ≥95% internal traffic uses DBE
- [ ] HTTP fallback <5% usage
- [ ] All message types (0x10-0x63) exchanged via DBE

### Performance
- [ ] DBE framing p99 < 50 μs
- [ ] PQC handshake p99 < 10 ms
- [ ] L7 round-trip p99 < 15 ms

### Security
- [ ] Tenant isolation enforced
- [ ] Kinetic compartment ban active
- [ ] ROE token validation for L9
- [ ] Two-person signatures for Device 61
- [ ] All 6 red-team tests passed

### Observability
- [ ] SHRINK monitoring DBE traffic
- [ ] Prometheus DBE metrics active
- [ ] Alerting configured for DBE errors

---

## 12. Complete Cryptographic Specification

This section provides the comprehensive cryptographic algorithm selection for all DSMIL use cases, ensuring consistency across the entire system.

### 12.1 Transport Layer (TLS/IPsec/SSH, DBE Protocol)

**Use Case:** Secure communication between DSMIL nodes, Layer 3-9 services

| Component | Algorithm | Key Size | Purpose |
|-----------|-----------|----------|---------|
| **Symmetric Encryption** | AES-256-GCM | 256-bit | Message confidentiality |
| **Key Derivation** | HKDF-SHA-384 | - | Session key derivation |
| **Key Exchange (PQC)** | ML-KEM-1024 | 1568 B | Post-quantum KEX |
| **Key Exchange (Classical)** | ECDH P-384 | 48 B | Hybrid KEX (transition) |
| **Authentication (PQC)** | ML-DSA-87 certificates | 4595 B | Node identity verification |
| **Authentication (Classical)** | ECDSA P-384 | 48 B | Hybrid auth (transition) |
| **Integrity** | SHA-384 HMAC | 384-bit | Message authentication |

**Implementation Notes:**
- Hybrid KEX: Combine ECDH P-384 + ML-KEM-1024 for transition period
- Hybrid Auth: Dual certificates (ML-DSA-87 + ECDSA P-384) during migration
- Phase out classical crypto once all nodes support PQC (target: 6 months post-deployment)

### 12.2 Data at Rest (Disk, Object Storage, Databases)

**Use Case:** Model weights (MLflow), tmpfs SQLite, Postgres warm storage, cold archive (S3/disk)

| Component | Algorithm | Key Size | Purpose |
|-----------|-----------|----------|---------|
| **Block Encryption** | AES-256-XTS | 256-bit (2× 128-bit keys) | Full-disk encryption |
| **Stream Encryption** | AES-256-CTR | 256-bit | Database column encryption |
| **Integrity** | AES-256-GCM (authenticated encryption) | 256-bit | File integrity verification |
| **Alternate Integrity** | SHA-384 HMAC | 384-bit | Large file checksums |
| **Key Encryption** | AES-256-GCM (KEK wrapping) | 256-bit | Database master key protection |

**Implementation Notes:**
- **Disk encryption:** AES-256-XTS for `/mnt/dsmil-ram/` tmpfs (if supported)
- **Database:** AES-256-CTR for Postgres Transparent Data Encryption (TDE)
- **Object storage:** AES-256-GCM for S3-compatible cold storage (server-side encryption)
- **Model weights:** AES-256-GCM via MLflow storage backend encryption
- **Integrity checks:** SHA-384 HMAC for large archives (> 1 GB); AES-GCM for smaller files

### 12.3 Firmware and OS Update Signing

**Use Case:** DSMIL software updates, kernel module signing, model package integrity

| Component | Algorithm | Key Size | Purpose |
|-----------|-----------|----------|---------|
| **Primary Signature (PQC)** | LMS (SHA-256/192) | - | Stateful hash-based signature |
| **Alternate (Stateless PQC)** | XMSS | - | Stateless hash-based (if HSM supports) |
| **Secondary Signature (Transition)** | ML-DSA-87 | 4595 B | Future-proof clients |
| **Classical (Legacy)** | RSA-4096 or ECDSA P-384 | - | Legacy compatibility |

**Implementation Notes:**
- **Preferred:** LMS (SHA-256/192) in HSM pipeline for firmware signing
  - Stateful, requires careful state management
  - NIST SP 800-208 compliant
  - Hardware acceleration available in TPM 2.0 and HSMs
- **Dual-sign strategy:**
  1. Primary: LMS signature (for PQC-ready systems)
  2. Secondary: ML-DSA-87 signature (for future clients)
  3. Legacy: ECDSA P-384 (for backward compatibility during transition)
- **Model package signing:**
  - MLflow packages signed with LMS + ML-DSA-87
  - Verification: Check both signatures (fail if either invalid)

### 12.4 Protocol-Internal Integrity and Nonce Derivation

**Use Case:** DBE protocol headers, sequence number integrity, nonce generation, internal checksums

| Component | Algorithm | Output Size | Purpose |
|-----------|-----------|-------------|---------|
| **Hash Function** | SHA-384 | 384-bit (48 B) | General-purpose hashing |
| **HMAC** | HMAC-SHA-384 | 384-bit (48 B) | Message authentication codes |
| **KDF** | HKDF-SHA-384 | Variable | All key derivation |
| **Nonce Derivation** | HKDF-SHA-384 | 96-bit (12 B) | AES-GCM nonce base |
| **Checksums** | SHA-384 | 384-bit (48 B) | File integrity checks |

**Implementation Notes:**
- **SHA-384 everywhere:** Default hash for all protocol-internal operations
- **No SHA-3:** Only use SHA-3-384/512 if hardware acceleration available AND you control the silicon
  - Intel Core Ultra 7 165H does NOT have SHA-3 acceleration → use SHA-384
- **HMAC-SHA-384:** For all message authentication (stronger than SHA-256 HMAC)
- **KDF standardization:** All key derivation uses HKDF-SHA-384 (no PBKDF2, no custom KDFs)

### 12.5 Quantum Cryptography (Device 61)

**Use Case:** Device 61 - Quantum Key Distribution (QKD) simulation

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| **Key Exchange (Simulated QKD)** | BB84 protocol (Qiskit) | Quantum key establishment |
| **Post-Processing** | Information reconciliation + privacy amplification | Classical post-QKD processing |
| **Key Storage** | AES-256-GCM wrapped keys | Derived quantum keys at rest |
| **Validation** | SHA-384 HMAC | Key authenticity verification |

**Implementation Notes:**
- Device 61 simulates QKD using Qiskit (no physical quantum channel)
- Generated quantum keys used for high-security Layer 9 operations
- Fallback: If QKD fails, use ML-KEM-1024 (same security level)

### 12.6 Legacy and Transition Period Support

**Algorithms supported during PQC migration (6-12 months):**

| Legacy Algorithm | Replacement | Transition Strategy |
|------------------|-------------|---------------------|
| RSA-2048/4096 | ML-DSA-87 | Dual-verify: accept both, prefer ML-DSA |
| ECDHE P-256 | ML-KEM-1024 + ECDH P-384 | Hybrid KEX mandatory |
| ECDSA P-256 | ML-DSA-87 + ECDSA P-384 | Dual-sign all new certificates |
| SHA-256 | SHA-384 | SHA-256 acceptable for LMS only |
| AES-128-GCM | AES-256-GCM | Reject AES-128 for new connections |

**Phase-out schedule:**
- **Month 0-3:** Hybrid mode (PQC + classical)
- **Month 3-6:** PQC preferred (classical warnings logged)
- **Month 6+:** PQC only (classical rejected except LMS)

### 12.7 Cryptographic Library Dependencies

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **liboqs** | ≥ 0.9.0 | ML-KEM-1024, ML-DSA-87, LMS | `apt install liboqs-dev` or build from source |
| **OpenSSL** | ≥ 3.2 | AES-GCM, SHA-384, ECDH/ECDSA, HKDF | `apt install openssl libssl-dev` |
| **OQS-OpenSSL Provider** | ≥ 0.6.0 | OpenSSL integration for PQC | Build from source |
| **Qiskit** | ≥ 1.0 | Quantum simulation (Device 46/61) | `pip install qiskit qiskit-aer` |

**Verification:**
```bash
# Check liboqs version
oqs-test --version

# Check OpenSSL PQC support
openssl list -providers | grep oqsprovider

# Test ML-KEM-1024
openssl pkey -in test_key.pem -text -noout | grep "ML-KEM"
```

---

## 13. Metadata

**Dependencies:**
- Phase 6 (External API Plane)
- liboqs 0.9+
- Rust 1.75+
- PyO3 0.20+

**Success Metrics:**
- 6× latency reduction (78ms → 12ms for L7)
- 100% high-classification traffic over PQC
- Zero kinetic compartment violations
- NC3 operations 100% two-person gated

**Next Phase:** Phase 8 (Advanced Analytics & ML Pipeline Hardening)

---

**Version History:**
- v1.0 (2024-Q4): Initial outline
- v2.0 (2025-11-23): Full v3.1 alignment with libdbe implementation

---

**End of Phase 7 Document**
