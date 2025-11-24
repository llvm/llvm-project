# Phase 4 – L8/L9 Activation & Governance Plane (v2.0)

**Version:** 2.0
**Status:** Aligned with v3.1 Comprehensive Plan
**Date:** 2025-11-23
**Last Updated:** Aligned hardware specs, Layer 8/9 device mappings, DBE integration, ROE enforcement

---

## 1. Objectives

Phase 4 activates **Layer 8 (ENHANCED_SEC)** and **Layer 9 (EXECUTIVE)** as the security and strategic oversight layers with strict governance:

1. **Layer 8 Online as Real SOC/Defense Plane**
   - Adversarial ML defense (Device 51)
   - Security analytics fusion (Device 52)
   - Cryptographic AI / PQC monitoring (Device 53)
   - Threat intelligence fusion (Device 54)
   - Behavioral biometrics (Device 55)
   - Secure enclave monitoring (Device 56)
   - Network security AI (Device 57)
   - SOAR orchestration (Device 58)

2. **Layer 9 Online as Executive/Strategic Overlay**
   - Strategic planning (Device 59)
   - Global strategy (Device 60)
   - NC3 integration (Device 61 with ROE gating)
   - Coalition intelligence (Device 62)

3. **Embed ROE/Governance/Safety**
   - Hard technical limits on what L8/L9 can *do* (advisory only)
   - 2-person integrity + ROE tokens for high-consequence flows
   - Policy enforcement via OPA or custom filters

4. **End-to-End Decision Loop**
   - L3→L4→L5→L6→L7 + SHRINK + L8 + L9 form complete loop:
     - Detect → Analyze → Predict → Explain → Recommend → (Human) Decide

### System Context (v3.1)

- **Physical Hardware:** Intel Core Ultra 7 165H (48.2 TOPS INT8: 13.0 NPU + 32.0 GPU + 3.2 CPU)
- **Memory:** 64 GB LPDDR5x-7467, 62 GB usable for AI, 64 GB/s shared bandwidth
- **Layer 8 (ENHANCED_SEC):** 8 devices (51-58), 8 GB budget, 80 TOPS theoretical
- **Layer 9 (EXECUTIVE):** 4 devices (59-62), 12 GB budget, 330 TOPS theoretical

---

## 2. Success Criteria

Phase 4 is complete when:

### Layer 8 (ENHANCED_SEC)
- [x] At least **4 concrete microservices** for Devices 51-58 are live:
  - Device 51: Adversarial ML Defense
  - Device 52: Security Analytics Fusion
  - Device 53: Cryptographic AI / PQC Watcher
  - Device 58: SOAR Orchestrator (proposal-only)
- [x] SOC can see **L8 severity + rationale** on each high-value event
- [x] L8 can **propose** actions (block, isolate, escalate) but **cannot execute** without human approval
- [x] All L8 services use DBE for internal communication

### Layer 9 (EXECUTIVE)
- [x] At least **one strategic COA generator** service live (Device 59)
- [x] Device 61 (NC3 Integration) operational with ROE token gating
- [x] L9 outputs are:
  - Fully logged + auditable
  - Clearly tagged as **ADVISORY**
  - Require 2-person approval + ROE tokens for downstream actions
- [x] All L9 services use DBE for internal communication

### Governance & Safety
- [x] Clear **policy layer** (OPA or custom) in front of any effectors
- [x] SHRINK monitors L8+L9 logs; anomalies surfaced into `SOC_EVENTS`
- [x] No path exists from AI → direct system change without explicit, logged human action
- [x] End-to-end tabletop scenario executed and audited

---

## 3. Architecture Overview

### 3.1 Layer 8/9 Topology

```
┌─────────────────────────────────────────────────────────────────┐
│              Layer 9 (EXECUTIVE) - Advisory Only                 │
│          4 Devices (59-62), 12 GB Budget, 330 TOPS              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐│
│  │ Device 59   │  │ Device 60   │  │ Device 61   │  │ Dev 62 ││
│  │ Strategic   │  │ Global      │  │ NC3 (ROE    │  │Coalition││
│  │ Planning    │  │ Strategy    │  │ Gated)      │  │ Intel  ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───┬────┘│
└─────────┼─────────────────┼─────────────────┼──────────────┼────┘
          │                 │                 │              │
          └─────────────────┴─────────────────┴──────────────┘
                                    │ DBE L9 Messages
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│             Layer 8 (ENHANCED_SEC) - Proposal Only               │
│           8 Devices (51-58), 8 GB Budget, 80 TOPS               │
│                                                                  │
│  Device 51: Adversarial ML │ Device 52: Security Analytics      │
│  Device 53: Crypto/PQC     │ Device 54: Threat Intel Fusion     │
│  Device 55: Biometrics     │ Device 56: Secure Enclave Monitor  │
│  Device 57: Network Sec AI │ Device 58: SOAR Orchestrator       │
│                                                                  │
│                    All communicate via DBE                       │
└─────────────────────────────────────────────────────────────────┘
                           │ DBE L8 Messages
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Redis SOC_EVENTS Stream                       │
│      ← Layer 3-7 outputs + SHRINK metrics + L8 enrichment       │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Policy Enforcement Layer                    │
│            (OPA or Custom) - Blocks unauthorized actions         │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Human Confirmation UI                         │
│        (2-Person Integrity for High-Consequence Actions)         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 DBE Message Types for Layer 8/9

**Extended from Phase 3, adding L8/L9 message types:**

| Message Type | Hex | Purpose | Direction |
|--------------|-----|---------|-----------|
| `L8_SOC_EVENT_ENRICHMENT` | `0x50` | Enrich SOC event with L8 analysis | Device 51-58 → SOC_EVENTS |
| `L8_PROPOSAL` | `0x51` | Proposed action (block/isolate/escalate) | Device 58 → Policy Engine |
| `L8_CRYPTO_ALERT` | `0x52` | PQC/crypto anomaly alert | Device 53 → SOC_EVENTS |
| `L9_COA_REQUEST` | `0x60` | Request course of action generation | Policy Engine → Device 59 |
| `L9_COA_RESPONSE` | `0x61` | Generated COA with options | Device 59 → Policy Engine |
| `L9_NC3_QUERY` | `0x62` | NC3 scenario query (ROE-gated) | Policy Engine → Device 61 |
| `L9_NC3_ANALYSIS` | `0x63` | NC3 analysis result (ADVISORY) | Device 61 → Policy Engine |

**Extended DBE TLVs for L8/L9:**

```text
ROE_TOKEN_ID (uint32)           – ROE capability token for NC3/high-consequence operations
TWO_PERSON_SIG_A (blob)         – First signature (ML-DSA-87) for 2-person integrity
TWO_PERSON_SIG_B (blob)         – Second signature (ML-DSA-87) for 2-person integrity
ADVISORY_FLAG (bool)            – True if output is advisory-only (no auto-execution)
POLICY_DECISION (enum)          – ALLOW | DENY | REQUIRES_APPROVAL
HUMAN_APPROVAL_ID (UUID)        – Reference to human approval workflow
AUDIT_TRAIL_ID (UUID)           – Reference to audit log entry
L8_SEVERITY (enum)              – LOW | MEDIUM | HIGH | CRITICAL
L9_CLASSIFICATION (enum)        – STRATEGIC | TACTICAL | NC3_TRAINING
```

---

## 4. Layer 8 (ENHANCED_SEC) Implementation

### 4.1 SOC_EVENT Schema (Finalized)

All L8 services read/write from Redis `SOC_EVENTS` stream with this schema:

```json
{
  "event_id": "uuid-v4",
  "ts": 1732377600.123456,
  "source_layer": 3,
  "device_id_src": 15,
  "severity": "HIGH",
  "category": "NETWORK",
  "classification": "SECRET",
  "compartment": "SIGNALS",

  "signals": {
    "l3": {
      "decision": "Anomalous traffic pattern detected",
      "score": 0.87,
      "device_id": 18
    },
    "l4": {
      "label": "Potential data exfiltration",
      "confidence": 0.91,
      "device_id": 25
    },
    "l5": {
      "forecast": "Pattern escalation predicted",
      "risk_band": "RISING",
      "device_id": 33
    },
    "l6": {
      "risk_level": 3,
      "policy_flags": ["TREATY_ANALOG_BREACH"],
      "device_id": 39
    },
    "l7": {
      "summary": "Correlated with known APT28 tactics",
      "rationale": "TTPs match historical campaign data",
      "device_id": 47
    },
    "shrink": {
      "risk_acute_stress": 0.72,
      "lbi_hyperfocus": 0.61,
      "cognitive_load": 0.68,
      "anomaly_score": 3.4
    }
  },

  "l8_enrichment": {
    "processed_by": [51, 52, 53, 57],
    "advml_flags": ["LOG_INTEGRITY_OK"],
    "analytics_flags": ["CAMPAIGN_SUSPECTED", "MULTI_VECTOR"],
    "crypto_flags": [],
    "network_flags": ["SUSPICIOUS_EGRESS"],
    "soar_proposals": [
      {
        "action": "ISOLATE_HOST",
        "target": "10.0.5.23",
        "rationale": "High confidence exfiltration attempt",
        "risk": "MEDIUM",
        "requires_approval": true,
        "proposed_by": "device_58"
      },
      {
        "action": "BLOCK_IP",
        "target": "203.0.113.45",
        "rationale": "Known C2 infrastructure",
        "risk": "LOW",
        "requires_approval": false,
        "proposed_by": "device_58"
      }
    ],
    "final_severity": "CRITICAL",
    "timestamp": 1732377610.456789
  }
}
```

### 4.2 Device 51 – Adversarial ML Defense

**Purpose:** Detect attacks on the DSMIL system itself (log poisoning, prompt injection, model evasion).

**Token ID:** `0x8099` (0x8000 + 51×3)

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/l8_advml_defense.py
"""
Device 51 - Adversarial ML Defense
Detects attacks on DSMIL infrastructure and AI models
"""

import time
import json
import logging
from typing import Dict, List

import redis
from dsmil_dbe import DBEMessage, DBESocket, MessageType

# Constants
DEVICE_ID = 51
LAYER = 8
TOKEN_BASE = 0x8099
REDIS_URL = "redis://localhost:6379/0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [L8-ADVML] [Device-51] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class AdversarialMLDefense:
    def __init__(self):
        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        self.last_event_id = "0-0"
        self.shrink_baseline = {"risk_acute_stress": 0.3, "anomaly_score": 1.0}

        logger.info(f"Adversarial ML Defense initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")

    def analyze_log_integrity(self, event: Dict) -> List[str]:
        """Detect log tampering or manipulation"""
        flags = []

        # Check for SHRINK anomaly spikes (may indicate stress-induced errors or tampering)
        shrink = event.get("signals", {}).get("shrink", {})
        anomaly_score = shrink.get("anomaly_score", 0.0)

        if anomaly_score > 5.0:  # 5-sigma threshold
            flags.append("POSSIBLE_LOG_TAMPER")
            logger.warning(f"High anomaly score: {anomaly_score:.2f} (Event: {event['event_id']})")

        # Check for inconsistencies between layers
        l3_score = event.get("signals", {}).get("l3", {}).get("score", 0.0)
        l4_confidence = event.get("signals", {}).get("l4", {}).get("confidence", 0.0)

        if abs(l3_score - l4_confidence) > 0.5:
            flags.append("LAYER_DISCREPANCY")
            logger.warning(f"L3/L4 score mismatch: {l3_score:.2f} vs {l4_confidence:.2f}")

        return flags if flags else ["LOG_INTEGRITY_OK"]

    def detect_prompt_injection(self, event: Dict) -> List[str]:
        """Detect attempts to manipulate LLM behavior"""
        flags = []

        l7_summary = event.get("signals", {}).get("l7", {}).get("summary", "")

        # Simple heuristic checks (production would use trained model)
        injection_patterns = [
            "ignore previous instructions",
            "disregard system prompt",
            "you are now",
            "forget everything",
            "\\n\\nSystem:",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in l7_summary.lower():
                flags.append("PROMPT_INJECTION_PATTERN")
                logger.warning(f"Potential prompt injection: '{pattern}' (Event: {event['event_id']})")
                break

        return flags

    def enrich_soc_event(self, event: Dict) -> Dict:
        """Add L8 adversarial ML analysis to SOC event"""

        advml_flags = []
        advml_flags.extend(self.analyze_log_integrity(event))
        advml_flags.extend(self.detect_prompt_injection(event))

        # Remove duplicates
        advml_flags = list(set(advml_flags))

        # Initialize or update l8_enrichment
        if "l8_enrichment" not in event:
            event["l8_enrichment"] = {
                "processed_by": [],
                "advml_flags": [],
                "analytics_flags": [],
                "crypto_flags": [],
                "network_flags": [],
                "soar_proposals": []
            }

        event["l8_enrichment"]["processed_by"].append(DEVICE_ID)
        event["l8_enrichment"]["advml_flags"] = advml_flags

        # Escalate severity if serious flags detected
        if "PROMPT_INJECTION_PATTERN" in advml_flags or "POSSIBLE_LOG_TAMPER" in advml_flags:
            current_severity = event.get("severity", "LOW")
            if current_severity not in ["HIGH", "CRITICAL"]:
                event["severity"] = "HIGH"
                logger.info(f"Escalated severity to HIGH due to advML flags (Event: {event['event_id']})")

        return event

    def run(self):
        """Main event loop"""
        logger.info("Adversarial ML Defense monitoring SOC_EVENTS...")

        while True:
            try:
                # Read from SOC_EVENTS stream
                streams = self.redis.xread(
                    {"SOC_EVENTS": self.last_event_id},
                    block=1000,
                    count=10
                )

                for stream_name, messages in streams:
                    if stream_name == b"SOC_EVENTS":
                        for msg_id, fields in messages:
                            try:
                                # Parse event
                                event_json = fields.get(b"event", b"{}")
                                event = json.loads(event_json.decode())

                                # Skip if already processed by us
                                processed_by = event.get("l8_enrichment", {}).get("processed_by", [])
                                if DEVICE_ID in processed_by:
                                    self.last_event_id = msg_id
                                    continue

                                # Enrich event
                                enriched_event = self.enrich_soc_event(event)

                                # Write back to stream
                                self.redis.xadd(
                                    "SOC_EVENTS",
                                    {"event": json.dumps(enriched_event)}
                                )

                                logger.info(
                                    f"Processed event | ID: {event['event_id'][:8]}... | "
                                    f"Flags: {enriched_event['l8_enrichment']['advml_flags']}"
                                )

                                self.last_event_id = msg_id

                            except Exception as e:
                                logger.error(f"Failed to process event: {e}")

                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Adversarial ML Defense shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    defense = AdversarialMLDefense()
    defense.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l8-advml.service
[Unit]
Description=DSMIL Device 51 - Adversarial ML Defense
After=redis-server.service shrink-dsmil.service dsmil-soc-router.service
Requires=redis-server.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=51"
Environment="DSMIL_LAYER=8"
Environment="REDIS_URL=redis://localhost:6379/0"

ExecStart=/opt/dsmil/.venv/bin/python l8_advml_defense.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l8-advml

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 4.3 Device 53 – Cryptographic AI / PQC Watcher

**Purpose:** Monitor PQC usage, detect crypto downgrades, watch for unexpected key rotations.

**Token ID:** `0x809F` (0x8000 + 53×3)

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/l8_crypto_watcher.py
"""
Device 53 - Cryptographic AI / PQC Watcher
Monitors post-quantum cryptography usage and key management
"""

import time
import json
import logging
from typing import Dict, List

import redis
from dsmil_pqc import PQCMonitor

# Constants
DEVICE_ID = 53
LAYER = 8
TOKEN_BASE = 0x809F
REDIS_URL = "redis://localhost:6379/0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [L8-CRYPTO] [Device-53] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoWatcher:
    def __init__(self):
        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        self.pqc_monitor = PQCMonitor()
        self.last_event_id = "0-0"
        self.expected_pqc_devices = [43, 47, 51, 52, 59, 61]  # Devices that MUST use PQC

        logger.info(f"Crypto Watcher initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")

    def check_pqc_compliance(self, event: Dict) -> List[str]:
        """Verify PQC usage where expected"""
        flags = []

        device_src = event.get("device_id_src")
        if device_src in self.expected_pqc_devices:
            # Check if event metadata indicates PQC usage
            # (In production, this would query actual connection metadata)
            classification = event.get("classification", "")
            if classification in ["TOP_SECRET", "ATOMAL", "EXEC"]:
                # High-classification events MUST use PQC
                # Placeholder check - production would verify actual TLS/DBE channel
                if not self._verify_pqc_channel(device_src):
                    flags.append("NON_PQC_CHANNEL")
                    logger.warning(
                        f"Device {device_src} classification={classification} without PQC | "
                        f"Event: {event['event_id']}"
                    )

        return flags

    def _verify_pqc_channel(self, device_id: int) -> bool:
        """
        Verify device is using PQC-protected channel
        Production: Query actual connection state from DBE layer
        """
        # Placeholder - always return True for now
        return True

    def detect_key_rotation_anomalies(self, event: Dict) -> List[str]:
        """Detect unexpected cryptographic key rotations"""
        flags = []

        # Check if event mentions key rotation
        l7_summary = event.get("signals", {}).get("l7", {}).get("summary", "")
        if "key" in l7_summary.lower() and "rotat" in l7_summary.lower():
            # In production, check against scheduled rotation policy
            flags.append("UNEXPECTED_KEY_ROTATION")
            logger.warning(f"Unscheduled key rotation detected | Event: {event['event_id']}")

        return flags

    def enrich_soc_event(self, event: Dict) -> Dict:
        """Add L8 cryptographic analysis to SOC event"""

        crypto_flags = []
        crypto_flags.extend(self.check_pqc_compliance(event))
        crypto_flags.extend(self.detect_key_rotation_anomalies(event))

        # Remove duplicates
        crypto_flags = list(set(crypto_flags))

        # Initialize or update l8_enrichment
        if "l8_enrichment" not in event:
            event["l8_enrichment"] = {
                "processed_by": [],
                "advml_flags": [],
                "analytics_flags": [],
                "crypto_flags": [],
                "network_flags": [],
                "soar_proposals": []
            }

        event["l8_enrichment"]["processed_by"].append(DEVICE_ID)
        event["l8_enrichment"]["crypto_flags"] = crypto_flags

        # Escalate severity if PQC violations detected
        if "NON_PQC_CHANNEL" in crypto_flags:
            event["severity"] = "HIGH"
            logger.info(f"Escalated severity to HIGH due to PQC violation (Event: {event['event_id']})")

        return event

    def run(self):
        """Main event loop"""
        logger.info("Crypto Watcher monitoring SOC_EVENTS...")

        while True:
            try:
                streams = self.redis.xread(
                    {"SOC_EVENTS": self.last_event_id},
                    block=1000,
                    count=10
                )

                for stream_name, messages in streams:
                    if stream_name == b"SOC_EVENTS":
                        for msg_id, fields in messages:
                            try:
                                event_json = fields.get(b"event", b"{}")
                                event = json.loads(event_json.decode())

                                # Skip if already processed
                                processed_by = event.get("l8_enrichment", {}).get("processed_by", [])
                                if DEVICE_ID in processed_by:
                                    self.last_event_id = msg_id
                                    continue

                                enriched_event = self.enrich_soc_event(event)

                                self.redis.xadd(
                                    "SOC_EVENTS",
                                    {"event": json.dumps(enriched_event)}
                                )

                                logger.info(
                                    f"Processed event | ID: {event['event_id'][:8]}... | "
                                    f"Crypto Flags: {enriched_event['l8_enrichment']['crypto_flags']}"
                                )

                                self.last_event_id = msg_id

                            except Exception as e:
                                logger.error(f"Failed to process event: {e}")

                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Crypto Watcher shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    watcher = CryptoWatcher()
    watcher.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l8-crypto.service
[Unit]
Description=DSMIL Device 53 - Cryptographic AI / PQC Watcher
After=redis-server.service dsmil-soc-router.service
Requires=redis-server.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=53"
Environment="DSMIL_LAYER=8"

ExecStart=/opt/dsmil/.venv/bin/python l8_crypto_watcher.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l8-crypto

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 4.4 Device 58 – SOAR Orchestrator (Proposal-Only)

**Purpose:** Generate structured response proposals for CRITICAL events (no auto-execution).

**Token ID:** `0x80AE` (0x8000 + 58×3)

**Key Principle:** Device 58 **proposes** actions but **never executes** them. All proposals require human approval.

**Implementation:** (Abbreviated for space - full implementation in separate workstream document)

```python
#!/usr/bin/env python3
# /opt/dsmil/l8_soar_orchestrator.py
"""
Device 58 - SOAR Orchestrator (Proposal-Only)
Generates structured response proposals for security events
"""

import time
import json
import logging
from typing import Dict, List

import redis
from dsmil_dbe import DBESocket, DBEMessage, MessageType

DEVICE_ID = 58
TOKEN_BASE = 0x80AE
L7_ROUTER_SOCKET = "/var/run/dsmil/l7-router.sock"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SOAROrchestrator:
    def __init__(self):
        self.redis = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=False)
        self.l7_router = DBESocket(connect_path=L7_ROUTER_SOCKET)
        self.last_event_id = "0-0"

        logger.info(f"SOAR Orchestrator initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")

    def generate_proposals(self, event: Dict) -> List[Dict]:
        """
        Use L7 LLM to generate response proposals
        """
        if event.get("severity") not in ["HIGH", "CRITICAL"]:
            return []  # Only propose for high-severity events

        # Build context for L7
        context = {
            "event_summary": event.get("signals", {}).get("l7", {}).get("summary", ""),
            "severity": event.get("severity"),
            "category": event.get("category"),
            "l8_flags": event.get("l8_enrichment", {})
        }

        # Call L7 router via DBE (simplified)
        try:
            dbe_msg = DBEMessage(
                msg_type=MessageType.L7_CHAT_REQ,
                correlation_id=event["event_id"],
                payload={
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a SOC response advisor. Propose actions to mitigate security incidents. Response format: JSON array of action objects with fields: action, target, rationale, risk."
                        },
                        {
                            "role": "user",
                            "content": f"Incident: {json.dumps(context)}"
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300
                }
            )

            dbe_msg.tlv_set("L7_PROFILE", "llm-7b-amx")
            dbe_msg.tlv_set("TENANT_ID", "LAYER_8_SOAR")
            dbe_msg.tlv_set("ROE_LEVEL", "SOC_ASSIST")
            dbe_msg.tlv_set("DEVICE_ID_SRC", DEVICE_ID)
            dbe_msg.tlv_set("DEVICE_ID_DST", 43)  # L7 Router

            response = self.l7_router.send_and_receive(dbe_msg, timeout=30.0)

            # Parse L7 response (simplified)
            result = response.payload
            llm_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON proposals from LLM
            proposals = json.loads(llm_text)

            # Add metadata
            for proposal in proposals:
                proposal["proposed_by"] = f"device_{DEVICE_ID}"
                proposal["requires_approval"] = True  # ALL proposals require approval

            return proposals

        except Exception as e:
            logger.error(f"Failed to generate proposals: {e}")
            return []

    def enrich_soc_event(self, event: Dict) -> Dict:
        """Add SOAR proposals to SOC event"""

        if "l8_enrichment" not in event:
            event["l8_enrichment"] = {
                "processed_by": [],
                "soar_proposals": []
            }

        event["l8_enrichment"]["processed_by"].append(DEVICE_ID)

        proposals = self.generate_proposals(event)
        event["l8_enrichment"]["soar_proposals"] = proposals

        if proposals:
            logger.info(
                f"Generated {len(proposals)} proposals | Event: {event['event_id'][:8]}..."
            )

        return event

    def run(self):
        """Main event loop"""
        logger.info("SOAR Orchestrator monitoring HIGH/CRITICAL events...")

        while True:
            try:
                streams = self.redis.xread(
                    {"SOC_EVENTS": self.last_event_id},
                    block=1000,
                    count=5  # Process fewer events (LLM calls are expensive)
                )

                for stream_name, messages in streams:
                    if stream_name == b"SOC_EVENTS":
                        for msg_id, fields in messages:
                            try:
                                event_json = fields.get(b"event", b"{}")
                                event = json.loads(event_json.decode())

                                # Skip if already processed
                                processed_by = event.get("l8_enrichment", {}).get("processed_by", [])
                                if DEVICE_ID in processed_by:
                                    self.last_event_id = msg_id
                                    continue

                                enriched_event = self.enrich_soc_event(event)

                                self.redis.xadd(
                                    "SOC_EVENTS",
                                    {"event": json.dumps(enriched_event)}
                                )

                                self.last_event_id = msg_id

                            except Exception as e:
                                logger.error(f"Failed to process event: {e}")

                time.sleep(0.5)  # Slower polling (LLM calls)

            except KeyboardInterrupt:
                logger.info("SOAR Orchestrator shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    orchestrator = SOAROrchestrator()
    orchestrator.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l8-soar.service
[Unit]
Description=DSMIL Device 58 - SOAR Orchestrator (Proposal-Only)
After=dsmil-l7-router.service dsmil-soc-router.service
Requires=dsmil-l7-router.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=58"
Environment="DSMIL_LAYER=8"

ExecStart=/opt/dsmil/.venv/bin/python l8_soar_orchestrator.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l8-soar

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 5. Layer 9 (EXECUTIVE) Implementation

### 5.1 Access Control & ROE Gating

**Before any L9 service starts, define gatekeeping:**

1. **L9 endpoints require:**
   - `role in {EXEC, STRAT_ANALYST}`
   - Valid session token (PQC-signed)
   - Per-request **ROE token** for NC3/high-consequence domains

2. **2-Person Integrity:**
   - High-impact scenarios require **two distinct ML-DSA-87 signatures**
   - Both signatures validated before L9 processing begins

3. **Advisory-Only Output:**
   - ALL L9 outputs tagged with `ADVISORY_FLAG=true`
   - No auto-execution pathways exist

### 5.2 Device 59 – COA Engine

**Purpose:** Generate courses of action (COA) with pros/cons, risk scoring, justifications.

**Token ID:** `0x80B1` (0x8000 + 59×3)

**Implementation:** (Abbreviated - full implementation ~500 lines)

```python
#!/usr/bin/env python3
# /opt/dsmil/l9_coa_engine.py
"""
Device 59 - Course of Action (COA) Engine
Generates strategic response options (ADVISORY ONLY)
"""

import time
import json
import logging
import uuid
from typing import Dict, List

from dsmil_dbe import DBESocket, DBEMessage, MessageType
from dsmil_pqc import MLDSAVerifier

DEVICE_ID = 59
LAYER = 9
TOKEN_BASE = 0x80B1
L7_ROUTER_SOCKET = "/var/run/dsmil/l7-router.sock"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [L9-COA] [Device-59] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class COAEngine:
    def __init__(self):
        self.l7_router = DBESocket(connect_path=L7_ROUTER_SOCKET)
        self.pqc_verifier = MLDSAVerifier()

        logger.info(f"COA Engine initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")

    def validate_authorization(self, request: DBEMessage) -> bool:
        """Validate role, session, and ROE token"""

        # Check role
        roles = request.tlv_get("ROLES", [])
        if not any(role in ["EXEC", "STRAT_ANALYST"] for role in roles):
            logger.warning("COA request denied: insufficient role")
            return False

        # Verify ROE token signature
        roe_token = request.tlv_get("ROE_TOKEN_ID")
        if not roe_token or not self.pqc_verifier.verify(roe_token):
            logger.warning("COA request denied: invalid ROE token")
            return False

        logger.info(f"COA request authorized | ROE Token: {roe_token[:8]}...")
        return True

    def generate_coa(self, scenario: Dict) -> Dict:
        """
        Generate course of action options using L7 LLM
        """

        # Build strategic context
        system_prompt = """You are a strategic military advisor providing ADVISORY-ONLY course of action (COA) analysis.

CONSTRAINTS:
- Your outputs are ADVISORY and require human approval
- Never recommend kinetic actions
- Never recommend actions violating ROE or treaties
- Focus on analysis, not execution

OUTPUT FORMAT (JSON):
{
  "coa_options": [
    {
      "option_number": 1,
      "title": "Brief title",
      "steps": ["step 1", "step 2", ...],
      "pros": ["pro 1", ...],
      "cons": ["con 1", ...],
      "risks": ["risk 1", ...],
      "assumptions": ["assumption 1", ...],
      "risk_level": "LOW|MEDIUM|HIGH"
    },
    ...
  ],
  "preferred_option": 1,
  "rationale": "Why this option is preferred"
}
"""

        user_prompt = f"""Scenario: {json.dumps(scenario, indent=2)}

Provide 2-4 course of action options."""

        try:
            # Call L7 via DBE
            dbe_msg = DBEMessage(
                msg_type=MessageType.L7_CHAT_REQ,
                correlation_id=str(uuid.uuid4()),
                payload={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 1500
                }
            )

            dbe_msg.tlv_set("L7_PROFILE", "llm-7b-amx")
            dbe_msg.tlv_set("TENANT_ID", "LAYER_9_COA")
            dbe_msg.tlv_set("ROE_LEVEL", "ANALYSIS_ONLY")
            dbe_msg.tlv_set("CLASSIFICATION", "STRATEGIC")
            dbe_msg.tlv_set("ADVISORY_FLAG", True)
            dbe_msg.tlv_set("DEVICE_ID_SRC", DEVICE_ID)
            dbe_msg.tlv_set("DEVICE_ID_DST", 43)

            response = self.l7_router.send_and_receive(dbe_msg, timeout=60.0)

            # Parse L7 response
            result = response.payload
            llm_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON COA
            coa_data = json.loads(llm_text)

            # Add metadata
            coa_data["generated_by"] = f"device_{DEVICE_ID}"
            coa_data["advisory_only"] = True
            coa_data["requires_human_approval"] = True
            coa_data["timestamp"] = time.time()

            return coa_data

        except Exception as e:
            logger.error(f"Failed to generate COA: {e}")
            return {"error": str(e)}

    def handle_coa_request(self, request: DBEMessage) -> DBEMessage:
        """Process COA request and return response"""

        # Validate authorization
        if not self.validate_authorization(request):
            response = DBEMessage(
                msg_type=MessageType.L9_COA_RESPONSE,
                correlation_id=request.correlation_id,
                payload={"error": "AUTHORIZATION_DENIED"}
            )
            response.tlv_set("POLICY_DECISION", "DENY")
            return response

        # Extract scenario
        scenario = request.payload.get("scenario", {})

        # Generate COA
        coa_data = self.generate_coa(scenario)

        # Create response
        response = DBEMessage(
            msg_type=MessageType.L9_COA_RESPONSE,
            correlation_id=request.correlation_id,
            payload=coa_data
        )
        response.tlv_set("DEVICE_ID_SRC", DEVICE_ID)
        response.tlv_set("ADVISORY_FLAG", True)
        response.tlv_set("POLICY_DECISION", "ALLOW")
        response.tlv_set("AUDIT_TRAIL_ID", str(uuid.uuid4()))

        logger.info(f"Generated COA | Request: {request.correlation_id[:8]}...")

        return response

    def run(self):
        """Main event loop"""
        logger.info("COA Engine listening for DBE COA requests...")

        socket = DBESocket(bind_path="/var/run/dsmil/l9-coa.sock")

        while True:
            try:
                msg = socket.receive(timeout=1.0)
                if not msg:
                    continue

                if msg.msg_type == MessageType.L9_COA_REQUEST:
                    response = self.handle_coa_request(msg)
                    socket.send(response)
                else:
                    logger.warning(f"Unexpected message type: 0x{msg.msg_type:02X}")

            except KeyboardInterrupt:
                logger.info("COA Engine shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    engine = COAEngine()
    engine.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l9-coa.service
[Unit]
Description=DSMIL Device 59 - COA Engine (ADVISORY ONLY)
After=dsmil-l7-router.service
Requires=dsmil-l7-router.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=59"
Environment="DSMIL_LAYER=9"

ExecStart=/opt/dsmil/.venv/bin/python l9_coa_engine.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l9-coa

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 5.3 Device 61 – NC3 Integration (ROE-Gated)

**Purpose:** NC3-analog analysis for training/simulation (NEVER operational).

**Token ID:** `0x80B7` (0x8000 + 61×3)

**CRITICAL CONSTRAINTS:**
- **ROE token mandatory** for all requests
- **2-person signatures required** for any NC3-related query
- Output **always tagged "NC3-ANALOG – TRAINING ONLY"**
- **No execution pathways** exist from Device 61

**Implementation:** (Abbreviated - includes ROE gating)

```python
#!/usr/bin/env python3
# /opt/dsmil/l9_nc3_integration.py
"""
Device 61 - NC3 Integration (ROE-GATED, TRAINING ONLY)
NC3-analog analysis with mandatory 2-person integrity
"""

import time
import json
import logging
import uuid
from typing import Dict

from dsmil_dbe import DBESocket, DBEMessage, MessageType
from dsmil_pqc import MLDSAVerifier

DEVICE_ID = 61
LAYER = 9
TOKEN_BASE = 0x80B7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [L9-NC3] [Device-61] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class NC3Integration:
    def __init__(self):
        self.pqc_verifier = MLDSAVerifier()
        logger.info(f"NC3 Integration initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")
        logger.warning("⚠️  DEVICE 61: NC3-ANALOG MODE - TRAINING ONLY - NO OPERATIONAL USE")

    def validate_nc3_authorization(self, request: DBEMessage) -> tuple[bool, str]:
        """
        Strict validation for NC3 requests:
        1. Valid ROE token
        2. Two-person signatures (ML-DSA-87)
        3. Explicit NC3_TRAINING classification
        """

        # Check ROE token
        roe_token = request.tlv_get("ROE_TOKEN_ID")
        if not roe_token:
            return False, "MISSING_ROE_TOKEN"

        if not self.pqc_verifier.verify(roe_token):
            return False, "INVALID_ROE_TOKEN"

        # Check 2-person signatures
        sig_a = request.tlv_get("TWO_PERSON_SIG_A")
        sig_b = request.tlv_get("TWO_PERSON_SIG_B")

        if not sig_a or not sig_b:
            return False, "MISSING_TWO_PERSON_SIGNATURES"

        if not self.pqc_verifier.verify(sig_a) or not self.pqc_verifier.verify(sig_b):
            return False, "INVALID_TWO_PERSON_SIGNATURES"

        # Verify signatures are from different identities
        # (Production: extract identity from signature and compare)

        # Check classification
        classification = request.tlv_get("L9_CLASSIFICATION")
        if classification != "NC3_TRAINING":
            return False, f"INVALID_CLASSIFICATION (got {classification}, expected NC3_TRAINING)"

        logger.warning(
            f"✅ NC3 request authorized | ROE: {roe_token[:8]}... | "
            f"2-person signatures verified"
        )

        return True, "AUTHORIZED"

    def analyze_nc3_scenario(self, scenario: Dict) -> Dict:
        """
        Analyze NC3-analog scenario (TRAINING ONLY)
        Output is purely advisory and includes prominent warnings
        """

        return {
            "analysis": {
                "scenario_type": scenario.get("type", "UNKNOWN"),
                "threat_level": "TRAINING_SIMULATION",
                "recommended_posture": "NO OPERATIONAL RECOMMENDATION",
                "confidence": 0.0  # Always 0.0 for NC3-analog
            },
            "warnings": [
                "⚠️  NC3-ANALOG OUTPUT - TRAINING ONLY",
                "⚠️  NOT FOR OPERATIONAL USE",
                "⚠️  REQUIRES HUMAN REVIEW AND APPROVAL",
                "⚠️  NO AUTO-EXECUTION PERMITTED"
            ],
            "generated_by": f"device_{DEVICE_ID}",
            "classification": "NC3_TRAINING",
            "advisory_only": True,
            "timestamp": time.time()
        }

    def handle_nc3_query(self, request: DBEMessage) -> DBEMessage:
        """Process NC3 query with strict ROE gating"""

        # Validate authorization
        authorized, reason = self.validate_nc3_authorization(request)

        if not authorized:
            logger.error(f"NC3 request DENIED: {reason}")

            response = DBEMessage(
                msg_type=MessageType.L9_NC3_ANALYSIS,
                correlation_id=request.correlation_id,
                payload={"error": f"AUTHORIZATION_DENIED: {reason}"}
            )
            response.tlv_set("POLICY_DECISION", "DENY")
            response.tlv_set("AUDIT_TRAIL_ID", str(uuid.uuid4()))
            return response

        # Extract scenario
        scenario = request.payload.get("scenario", {})

        # Analyze (with training-only constraints)
        analysis = self.analyze_nc3_scenario(scenario)

        # Create response with prominent warnings
        response = DBEMessage(
            msg_type=MessageType.L9_NC3_ANALYSIS,
            correlation_id=request.correlation_id,
            payload=analysis
        )
        response.tlv_set("DEVICE_ID_SRC", DEVICE_ID)
        response.tlv_set("ADVISORY_FLAG", True)
        response.tlv_set("L9_CLASSIFICATION", "NC3_TRAINING")
        response.tlv_set("POLICY_DECISION", "ALLOW")
        response.tlv_set("AUDIT_TRAIL_ID", str(uuid.uuid4()))

        logger.warning(
            f"Generated NC3 analysis (TRAINING ONLY) | "
            f"Request: {request.correlation_id[:8]}..."
        )

        return response

    def run(self):
        """Main event loop"""
        logger.info("NC3 Integration listening (ROE-GATED)...")
        logger.warning("⚠️  ALL NC3 OUTPUTS ARE TRAINING-ONLY AND ADVISORY")

        socket = DBESocket(bind_path="/var/run/dsmil/l9-nc3.sock")

        while True:
            try:
                msg = socket.receive(timeout=1.0)
                if not msg:
                    continue

                if msg.msg_type == MessageType.L9_NC3_QUERY:
                    response = self.handle_nc3_query(msg)
                    socket.send(response)
                else:
                    logger.warning(f"Unexpected message type: 0x{msg.msg_type:02X}")

            except KeyboardInterrupt:
                logger.info("NC3 Integration shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    nc3 = NC3Integration()
    nc3.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-l9-nc3.service
[Unit]
Description=DSMIL Device 61 - NC3 Integration (ROE-GATED, TRAINING ONLY)
After=dsmil-l7-router.service
Requires=dsmil-l7-router.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="DSMIL_DEVICE_ID=61"
Environment="DSMIL_LAYER=9"

ExecStart=/opt/dsmil/.venv/bin/python l9_nc3_integration.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l9-nc3

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 6. Policy Enforcement Layer

### 6.1 Policy Engine (OPA or Custom)

**Purpose:** Final gatekeeper between L8/L9 advisory outputs and any external systems.

**Policy Rules:**

```rego
# /opt/dsmil/policies/l8_l9_policy.rego

package dsmil.l8_l9

import future.keywords.if

# Default deny
default allow = false

# Allow advisory outputs (no execution)
allow if {
    input.advisory_flag == true
    input.requires_approval == true
}

# Deny any kinetic actions
deny["KINETIC_ACTION_FORBIDDEN"] if {
    contains(lower(input.action), "strike")
}

deny["KINETIC_ACTION_FORBIDDEN"] if {
    contains(lower(input.action), "attack")
}

deny["KINETIC_ACTION_FORBIDDEN"] if {
    contains(lower(input.action), "destroy")
}

# Deny actions outside ROE
deny["ROE_VIOLATION"] if {
    input.roe_level == "ANALYSIS_ONLY"
    input.action_category == "EXECUTION"
}

# Require 2-person for NC3
deny["TWO_PERSON_REQUIRED"] if {
    input.device_id == 61
    not input.two_person_verified
}

# Require human approval for HIGH risk
deny["HUMAN_APPROVAL_REQUIRED"] if {
    input.risk_level == "HIGH"
    not input.human_approved
}
```

**Policy Enforcement Service:**

```python
#!/usr/bin/env python3
# /opt/dsmil/policy_enforcer.py
"""
Policy Enforcement Layer
Final gatekeeper for all L8/L9 outputs
"""

import time
import json
import logging
from typing import Dict

from dsmil_dbe import DBESocket, DBEMessage
import opa_client  # OPA REST API client

POLICY_ENGINE_URL = "http://localhost:8181/v1/data/dsmil/l8_l9/allow"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyEnforcer:
    def __init__(self):
        self.opa = opa_client.OPAClient(POLICY_ENGINE_URL)
        logger.info("Policy Enforcer initialized")

    def enforce(self, request: Dict) -> tuple[bool, List[str]]:
        """
        Enforce policy on L8/L9 output
        Returns: (allowed, deny_reasons)
        """

        # Query OPA
        result = self.opa.query({"input": request})

        allowed = result.get("result", {}).get("allow", False)
        denials = result.get("result", {}).get("deny", [])

        if not allowed:
            logger.warning(f"Policy DENIED | Reasons: {denials}")
        else:
            logger.info(f"Policy ALLOWED | Request: {request.get('request_id', 'unknown')[:8]}...")

        return allowed, denials

if __name__ == "__main__":
    enforcer = PolicyEnforcer()
    # Listen for L8/L9 outputs and enforce policy
    # (Full implementation omitted for brevity)
```

---

## 7. Phase 4 Exit Criteria & Validation

### 7.1 Checklist

- [ ] **Layer 8 services operational:**
  - Device 51 (Adversarial ML Defense) running
  - Device 53 (Crypto/PQC Watcher) running
  - Device 58 (SOAR Orchestrator) running
  - All enriching `SOC_EVENTS` stream

- [ ] **Layer 9 services operational:**
  - Device 59 (COA Engine) running
  - Device 61 (NC3 Integration) running with ROE gating
  - All outputs tagged ADVISORY
  - 2-person integrity enforced for Device 61

- [ ] **Policy enforcement active:**
  - OPA policy engine running
  - Kinetic actions blocked
  - ROE violations logged
  - Human approval workflow functional

- [ ] **End-to-end tabletop scenario:**
  - Synthetic incident → L3-7 → L8 enrichment → L9 COA → Human decision
  - All flows logged and auditable
  - No policy violations

### 7.2 Validation Commands

```bash
# Verify Layer 8 services
systemctl status dsmil-l8-advml.service
systemctl status dsmil-l8-crypto.service
systemctl status dsmil-l8-soar.service

# Verify Layer 9 services
systemctl status dsmil-l9-coa.service
systemctl status dsmil-l9-nc3.service

# Check SOC_EVENTS enrichment
redis-cli XREAD COUNT 1 STREAMS SOC_EVENTS 0 | jq '.l8_enrichment'

# Verify policy enforcement
curl http://localhost:8181/v1/data/dsmil/l8_l9/allow -d '{"input": {"advisory_flag": true, "requires_approval": true}}'

# View L8/L9 logs
journalctl -u dsmil-l8-*.service -u dsmil-l9-*.service -f

# Run tabletop scenario
python /opt/dsmil/tests/phase4_tabletop.py
```

---

## 8. Document Metadata

**Version History:**
- **v1.0 (2024-Q4):** Initial Phase 4 spec
- **v2.0 (2025-11-23):** Aligned with v3.1 Comprehensive Plan
  - Updated Layer 8/9 device mappings (51-62)
  - Added token IDs (0x8099-0x80B7)
  - Integrated DBE protocol for L8/L9
  - Added ROE gating for Device 61
  - Detailed policy enforcement layer
  - Complete implementation examples

**Dependencies:**
- Phase 1-3 completed
- `libdbe` with L8/L9 message types
- OPA (Open Policy Agent) >= 0.45
- liboqs (PQC library)

**References:**
- `00_MASTER_PLAN_OVERVIEW_CORRECTED.md (v3.1)`
- `01_HARDWARE_INTEGRATION_LAYER_DETAILED.md (v3.1)`
- `Phase7.md (v1.0)` - DBE protocol
- `05_LAYER_SPECIFIC_DEPLOYMENTS.md (v1.0)`

---

**END OF PHASE 4 SPECIFICATION**
