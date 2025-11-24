# Cross-Layer Intelligence Flows & Orchestration

**Version**: 1.0
**Date**: 2025-11-23
**Status**: Design Complete – Implementation Ready
**Project**: DSMIL AI System Integration

---

## Executive Summary

This document specifies **cross-layer intelligence flows** and **orchestration patterns** for the complete DSMIL 104-device, 9-layer architecture.

**Key Principles**:

1. **Upward Intelligence Flow**: Lower layers push intelligence upward; higher layers never query down directly
2. **Security Boundaries**: Each layer enforces clearance checks; data crosses boundaries only with authorization
3. **Device Orchestration**: 104 devices coordinate via the Hardware Integration Layer (HIL)
4. **DIRECTEYE Integration**: 35+ specialized tools interface with DSMIL devices for multi-modal intelligence
5. **Event-Driven Architecture**: Devices publish events; higher layers subscribe with clearance verification

**Flow Hierarchy**:

```text
Layer 9 (EXECUTIVE) ← Global synthesis
    ↑
Layer 8 (ENHANCED_SEC) ← Security overlay
    ↑
Layer 7 (EXTENDED) ← PRIMARY AI/ML synthesis
    ↑
Layer 6 (ATOMAL) ← Nuclear intelligence
    ↑
Layer 5 (COSMIC) ← Predictive analytics
    ↑
Layer 4 (TOP_SECRET) ← Mission planning
    ↑
Layer 3 (SECRET) ← Domain analytics
    ↑
Layer 2 (TRAINING) ← Development (isolated)
```

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Intelligence Flow Patterns](#2-intelligence-flow-patterns)
3. [Cross-Layer Data Routing](#3-cross-layer-data-routing)
4. [Device Orchestration](#4-device-orchestration)
5. [Security Enforcement](#5-security-enforcement)
6. [DIRECTEYE Integration](#6-directeye-integration)
7. [Event-Driven Intelligence](#7-event-driven-intelligence)
8. [Workflow Examples](#8-workflow-examples)
9. [Performance & Optimization](#9-performance--optimization)
10. [Implementation](#10-implementation)

---

## 1. Architecture Overview

### 1.1 Multi-Layer Intelligence Stack

```text
┌──────────────────────────────────────────────────────────────────┐
│               DSMIL Cross-Layer Intelligence Stack               │
│          104 Devices, 9 Operational Layers, Event-Driven         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Layer 9 (EXECUTIVE) – 4 devices                                  │
│   Global Synthesis | Executive Command | NC3 | Coalition         │
│   ↑ Subscribes to: Layers 7, 8 (strategic intelligence)          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 8 (ENHANCED_SEC) – 8 devices                               │
│   Security AI | PQC | Zero-Trust | Deepfake Detection            │
│   ↑ Subscribes to: All layers (security monitoring)              │
│   → Provides: Security overlay for all layers                    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 7 (EXTENDED) – 8 devices ★ PRIMARY AI/ML                   │
│   Advanced AI/ML (Device 47 LLM) | Quantum | Strategic | OSINT   │
│   ↑ Subscribes to: Layers 2–6 (all intelligence feeds)           │
│   → Provides: High-level synthesis, strategic reasoning          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 6 (ATOMAL) – 6 devices                                     │
│   Nuclear Intelligence | NC3 | Treaty Monitoring                 │
│   ↑ Subscribes to: Layers 3–5 (nuclear-relevant intelligence)    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5 (COSMIC) – 6 devices                                     │
│   Predictive Analytics | Coalition Intel | Geospatial            │
│   ↑ Subscribes to: Layers 3–4 (mission + domain data)            │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4 (TOP_SECRET) – 8 devices                                 │
│   Mission Planning | Intel Fusion | Risk Assessment              │
│   ↑ Subscribes to: Layer 3 (domain analytics)                    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3 (SECRET) – 8 devices                                     │
│   CRYPTO | SIGNALS | NUCLEAR | WEAPONS | COMMS | etc.            │
│   ↑ Subscribes to: Raw sensor/data feeds (Layer 0 system devices)│
├──────────────────────────────────────────────────────────────────┤
│ Layer 2 (TRAINING) – 1 device                                    │
│   Development/Testing (isolated, no production feeds)            │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴────────────────────────────────────┐
│         Hardware Integration Layer (HIL) – Orchestration         │
│   Device Token Routing | Memory Management | Security Gates      │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Principles

**1. Upward-Only Intelligence Flow**:
- Layer N can subscribe to events from Layers < N
- Layer N **cannot** query Layers > N
- Enforced via token-based access control at HIL

**2. Event-Driven Architecture**:
- Devices publish events (intelligence products) to HIL event bus
- Higher-layer devices subscribe with clearance verification
- Asynchronous, non-blocking (no direct device-to-device calls)

**3. Security Boundaries**:
- Each layer transition requires clearance check
- Layer 8 (ENHANCED_SEC) monitors all cross-layer flows
- Audit logging at every boundary crossing

**4. Layer 7 as Synthesis Hub**:
- Layer 7 (Device 47 LLM) synthesizes intelligence from Layers 2–6
- Acts as "reasoning engine" before executive layer
- 40 GB memory budget supports multi-source fusion

---

## 2. Intelligence Flow Patterns

### 2.1 Flow Types

**Type 1: Raw Sensor Data → Domain Analytics (Layer 3)**

```text
System Devices (0–11) → Layer 3 Devices (15–22)

Example:
Device 5 (Network Interface) → Device 16 (SIGNALS)
   Raw RF intercepts → Signal classification
```

**Type 2: Domain Analytics → Mission Planning (Layer 3 → 4)**

```text
Layer 3 Devices (15–22) → Layer 4 Devices (23–30)

Example:
Device 18 (WEAPONS) → Device 23 (Mission Planning)
   Weapon signature detection → Mission threat assessment
```

**Type 3: Mission Planning → Predictive Analytics (Layer 4 → 5)**

```text
Layer 4 Devices (23–30) → Layer 5 Devices (31–36)

Example:
Device 25 (Intel Fusion) → Device 31 (Predictive Analytics)
   Fused intelligence → Strategic forecasting
```

**Type 4: Multi-Source → Layer 7 Synthesis (Layers 2–6 → 7)**

```text
All Lower Layers → Layer 7 Device 47 (Advanced AI/ML)

Example:
Device 16 (SIGNALS) + Device 25 (Intel Fusion) + Device 31 (Predictive)
   → Device 47 (LLM) → Comprehensive strategic assessment
```

**Type 5: Strategic Intelligence → Executive Command (Layer 7 → 9)**

```text
Layer 7 Devices (43–50) → Layer 9 Devices (59–62)

Example:
Device 47 (Advanced AI/ML) → Device 59 (Executive Command)
   Strategic COAs → Executive decision recommendation
```

**Type 6: Security Overlay (Layer 8 ↔ All Layers)**

```text
Layer 8 Devices (51–58) ↔ All Layers (bidirectional monitoring)

Example:
Device 52 (Security AI) monitors all layer transitions
   → Detects anomalous cross-layer queries
   → Triggers Device 83 (Emergency Stop) if breach detected
```

### 2.2 Flow Latency Budgets

| Flow Type | Layers | Latency Budget | Priority |
|-----------|--------|----------------|----------|
| Type 1 | System → 3 | < 100 ms | HIGH (real-time sensors) |
| Type 2 | 3 → 4 | < 500 ms | MEDIUM (mission-relevant) |
| Type 3 | 4 → 5 | < 1 sec | MEDIUM |
| Type 4 | 2–6 → 7 | < 2 sec | HIGH (synthesis critical) |
| Type 5 | 7 → 9 | < 1 sec | CRITICAL (executive) |
| Type 6 | 8 ↔ All | < 50 ms | CRITICAL (security) |

---

## 3. Cross-Layer Data Routing

### 3.1 Token-Based Routing

**Device Token Format**:
```python
TOKEN_ID = 0x8000 + (device_id × 3) + offset
# offset: 0=STATUS, 1=CONFIG, 2=DATA
```

**Cross-Layer Query Example**:

```python
# Layer 7 Device 47 queries Layer 3 Device 16 (SIGNALS)
SOURCE_DEVICE = 47  # Layer 7
TARGET_DEVICE = 16  # Layer 3
QUERY_TOKEN = 0x8000 + (16 × 3) + 2  # 0x8000 + 48 + 2 = 0x8032 (DATA)

# Clearance check
SOURCE_CLEARANCE = 0x07070707  # Layer 7 (EXTENDED)
TARGET_CLEARANCE = 0x03030303  # Layer 3 (SECRET)

# Authorization: Layer 7 ≥ Layer 3 → ALLOWED (upward query)
# If SOURCE_CLEARANCE < TARGET_CLEARANCE → DENIED
```

### 3.2 Routing Enforcement

**Hardware Integration Layer (HIL) Router**:

```python
class CrossLayerRouter:
    """
    Enforces upward-only intelligence flow with clearance checks.
    """

    DEVICE_LAYER_MAP = {
        # System devices
        **{i: 0 for i in range(0, 12)},
        # Security devices
        **{i: 0 for i in range(12, 15)},
        # Layer 3 (SECRET)
        **{i: 3 for i in range(15, 23)},
        # Layer 4 (TOP_SECRET)
        **{i: 4 for i in range(23, 31)},
        # Layer 5 (COSMIC)
        **{i: 5 for i in range(31, 37)},
        # Layer 6 (ATOMAL)
        **{i: 6 for i in range(37, 43)},
        # Layer 7 (EXTENDED)
        **{i: 7 for i in range(43, 51)},
        # Layer 8 (ENHANCED_SEC)
        **{i: 8 for i in range(51, 59)},
        # Layer 9 (EXECUTIVE)
        **{i: 9 for i in range(59, 63)},
        # Reserved
        **{i: 0 for i in range(63, 104)},
    }

    LAYER_CLEARANCES = {
        2: 0x02020202,
        3: 0x03030303,
        4: 0x04040404,
        5: 0x05050505,
        6: 0x06060606,
        7: 0x07070707,
        8: 0x08080808,
        9: 0x09090909,
    }

    def authorize_query(self, source_device_id: int, target_device_id: int) -> bool:
        """
        Authorize cross-layer query.

        Rules:
        - Source layer ≥ Target layer: ALLOWED (upward query)
        - Source layer < Target layer: DENIED (downward query blocked)
        - Layer 8 (ENHANCED_SEC): ALLOWED to query any layer (security monitoring)
        - Device 83 (Emergency Stop): ALLOWED to halt any device
        """
        source_layer = self.DEVICE_LAYER_MAP.get(source_device_id, 0)
        target_layer = self.DEVICE_LAYER_MAP.get(target_device_id, 0)

        # Special cases
        if source_device_id == 83:  # Emergency Stop
            return True
        if source_layer == 8:  # Layer 8 can monitor all
            return True

        # Standard upward-only rule
        if source_layer >= target_layer:
            return True

        # Deny downward queries
        return False

    def route_intelligence(
        self,
        source_device_id: int,
        target_device_id: int,
        data: bytes,
        metadata: dict
    ) -> tuple[bool, str]:
        """
        Route intelligence between devices with authorization and audit.
        """
        # Authorization check
        if not self.authorize_query(source_device_id, target_device_id):
            audit_log = {
                "event": "CROSS_LAYER_QUERY_DENIED",
                "source_device": source_device_id,
                "target_device": target_device_id,
                "reason": "Downward query blocked (upward-only policy)",
                "timestamp": time.time(),
            }
            self.log_security_event(audit_log)
            return False, "Authorization denied"

        # Token-based delivery
        target_token = 0x8000 + (target_device_id * 3) + 2  # DATA token

        # Construct message
        message = {
            "source_device": source_device_id,
            "target_device": target_device_id,
            "token": target_token,
            "data": data,
            "metadata": metadata,
            "timestamp": time.time(),
        }

        # Deliver via HIL
        success = self.hil.send_message(target_token, message)

        # Audit log
        audit_log = {
            "event": "CROSS_LAYER_INTELLIGENCE_FLOW",
            "source_device": source_device_id,
            "target_device": target_device_id,
            "data_size_bytes": len(data),
            "success": success,
            "timestamp": time.time(),
        }
        self.log_audit(audit_log)

        return success, "Intelligence routed"
```

### 3.3 Routing Patterns

**Pattern 1: Fan-In (Multiple Sources → Single Sink)**

```text
Device 15 (CRYPTO)  ┐
Device 16 (SIGNALS) ├─→ Device 25 (Intel Fusion, Layer 4)
Device 17 (NUCLEAR) ┘

All Layer 3 devices feed into Layer 4 fusion device.
```

**Pattern 2: Fan-Out (Single Source → Multiple Sinks)**

```text
                    ┌─→ Device 31 (Predictive Analytics)
Device 25 (Intel   ├─→ Device 34 (Threat Assessment)
Fusion, Layer 4)   └─→ Device 37 (ATOMAL Fusion)

Single fusion output propagates to multiple Layer 5–6 devices.
```

**Pattern 3: Cascade (Sequential Layer Progression)**

```text
Device 16 (SIGNALS, Layer 3)
    ↓
Device 25 (Intel Fusion, Layer 4)
    ↓
Device 31 (Predictive Analytics, Layer 5)
    ↓
Device 47 (Advanced AI/ML, Layer 7)
    ↓
Device 59 (Executive Command, Layer 9)

Intelligence progressively refined through layers.
```

---

## 4. Device Orchestration

### 4.1 Orchestration Modes

**Mode 1: Pipeline (Sequential Processing)**

```python
pipeline = [
    {"device": 16, "operation": "signal_classification"},
    {"device": 25, "operation": "intel_fusion"},
    {"device": 47, "operation": "strategic_reasoning"},
    {"device": 59, "operation": "executive_recommendation"},
]

result = orchestrator.execute_pipeline(pipeline, input_data)
```

**Mode 2: Parallel (Concurrent Processing)**

```python
parallel_tasks = [
    {"device": 15, "operation": "crypto_analysis"},
    {"device": 16, "operation": "signal_analysis"},
    {"device": 17, "operation": "nuclear_analysis"},
]

results = orchestrator.execute_parallel(parallel_tasks, input_data)
fused = orchestrator.fuse_results(results, fusion_device=25)
```

**Mode 3: Event-Driven (Publish-Subscribe)**

```python
# Device 16 publishes event
event = {
    "device_id": 16,
    "event_type": "SIGNAL_DETECTED",
    "data": signal_data,
    "classification": "high_priority",
    "timestamp": time.time(),
}
orchestrator.publish_event(event)

# Devices 25, 31, 47 subscribe to "SIGNAL_DETECTED" events
# Each receives event asynchronously, processes independently
```

### 4.2 Orchestration API

```python
class DSMILOrchestrator:
    """
    104-device orchestration engine with cross-layer intelligence routing.
    """

    def __init__(self, hil: HardwareIntegrationLayer):
        self.hil = hil
        self.router = CrossLayerRouter(hil)
        self.event_bus = EventBus()

    def execute_pipeline(
        self,
        pipeline: list[dict],
        input_data: bytes
    ) -> dict:
        """
        Execute sequential pipeline across devices.
        """
        data = input_data
        results = []

        for step in pipeline:
            device_id = step["device"]
            operation = step["operation"]

            # Send to device
            token = 0x8000 + (device_id * 3) + 2  # DATA token
            response = self.hil.send_and_receive(token, {
                "operation": operation,
                "data": data,
            })

            # Collect result
            results.append(response)
            data = response["output"]  # Feed to next stage

        return {
            "pipeline_results": results,
            "final_output": data,
        }

    def execute_parallel(
        self,
        tasks: list[dict],
        input_data: bytes
    ) -> list[dict]:
        """
        Execute tasks concurrently across devices.
        """
        futures = []

        for task in tasks:
            device_id = task["device"]
            operation = task["operation"]
            token = 0x8000 + (device_id * 3) + 2

            # Async send
            future = self.hil.send_async(token, {
                "operation": operation,
                "data": input_data,
            })
            futures.append((device_id, future))

        # Wait for all
        results = []
        for device_id, future in futures:
            response = future.wait()
            results.append({
                "device_id": device_id,
                "result": response,
            })

        return results

    def publish_event(self, event: dict) -> None:
        """
        Publish event to event bus for subscriber devices.
        """
        self.event_bus.publish(event)

        # Audit log
        self.router.log_audit({
            "event": "INTELLIGENCE_EVENT_PUBLISHED",
            "source_device": event["device_id"],
            "event_type": event["event_type"],
            "timestamp": time.time(),
        })

    def subscribe_device(
        self,
        device_id: int,
        event_types: list[str],
        callback: callable
    ) -> None:
        """
        Subscribe device to event types.
        """
        for event_type in event_types:
            self.event_bus.subscribe(event_type, device_id, callback)
```

---

## 5. Security Enforcement

### 5.1 Clearance Verification

**Per-Query Clearance Check**:

```python
class SecurityGate:
    """
    Enforces clearance requirements for cross-layer intelligence flow.
    """

    def verify_clearance(
        self,
        source_device_id: int,
        target_device_id: int,
        user_clearance: int
    ) -> tuple[bool, str]:
        """
        Verify clearance for cross-layer query.

        Requirements:
        1. Source device layer ≥ Target device layer (upward-only)
        2. User clearance ≥ Target device layer clearance
        3. Layer 8 monitoring active (security overlay)
        """
        source_layer = self.router.DEVICE_LAYER_MAP[source_device_id]
        target_layer = self.router.DEVICE_LAYER_MAP[target_device_id]
        target_clearance = self.router.LAYER_CLEARANCES[target_layer]

        # Check 1: Upward-only (handled by router)
        if not self.router.authorize_query(source_device_id, target_device_id):
            return False, "Upward-only policy violation"

        # Check 2: User clearance
        if user_clearance < target_clearance:
            return False, f"Insufficient clearance: user={hex(user_clearance)}, required={hex(target_clearance)}"

        # Check 3: Layer 8 security monitoring
        if not self.layer8_monitoring_active():
            return False, "Security monitoring offline (Layer 8 required)"

        return True, "Clearance verified"

    def layer8_monitoring_active(self) -> bool:
        """
        Check if Layer 8 (ENHANCED_SEC) is actively monitoring.
        """
        # Check Device 52 (Security AI) status
        token = 0x8000 + (52 * 3) + 0  # STATUS token
        status = self.hil.query(token)
        return status["monitoring_active"]
```

### 5.2 Audit Logging

**Comprehensive Audit Trail**:

```python
class AuditLogger:
    """
    Logs all cross-layer intelligence flows for security audit.
    """

    def log_cross_layer_query(
        self,
        source_device_id: int,
        target_device_id: int,
        user_id: str,
        clearance: int,
        authorized: bool,
        data_size_bytes: int
    ) -> None:
        """
        Log cross-layer query with full context.
        """
        log_entry = {
            "timestamp": time.time(),
            "event_type": "CROSS_LAYER_QUERY",
            "source_device": source_device_id,
            "source_layer": self.router.DEVICE_LAYER_MAP[source_device_id],
            "target_device": target_device_id,
            "target_layer": self.router.DEVICE_LAYER_MAP[target_device_id],
            "user_id": user_id,
            "user_clearance": hex(clearance),
            "authorized": authorized,
            "data_size_bytes": data_size_bytes,
        }

        # Write to audit log (Device 14: Audit Logger)
        audit_token = 0x8000 + (14 * 3) + 2  # DATA token
        self.hil.send(audit_token, log_entry)

        # Also log to Layer 8 (Security AI)
        layer8_token = 0x8000 + (52 * 3) + 2
        self.hil.send(layer8_token, log_entry)
```

### 5.3 Emergency Stop (Device 83)

**Device 83: Hardware Read-Only Emergency Stop**

```python
class EmergencyStop:
    """
    Device 83: Emergency stop for security breaches.
    Hardware read-only; cannot be overridden by software.
    """

    DEVICE_ID = 83
    TOKEN_STATUS = 0x8000 + (83 * 3) + 0

    def trigger_emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop across all devices.

        Actions:
        1. Halt all device operations
        2. Freeze memory (no writes)
        3. Capture forensic snapshot
        4. Alert Layer 8 and Layer 9
        """
        # Send emergency halt to all devices
        for device_id in range(104):
            token = 0x8000 + (device_id * 3) + 1  # CONFIG token
            self.hil.send(token, {
                "command": "EMERGENCY_HALT",
                "reason": reason,
                "triggered_by": self.DEVICE_ID,
            })

        # Forensic snapshot
        self.capture_forensic_snapshot()

        # Alert Layer 8 (Security AI)
        layer8_token = 0x8000 + (52 * 3) + 2
        self.hil.send(layer8_token, {
            "event": "EMERGENCY_STOP_TRIGGERED",
            "reason": reason,
            "timestamp": time.time(),
        })

        # Alert Layer 9 (Executive Command)
        layer9_token = 0x8000 + (59 * 3) + 2
        self.hil.send(layer9_token, {
            "event": "EMERGENCY_STOP_TRIGGERED",
            "reason": reason,
            "timestamp": time.time(),
        })
```

---

## 6. DIRECTEYE Integration

### 6.1 DIRECTEYE Overview

**DIRECTEYE**: Specialized intelligence toolkit with **35+ tools** for multi-modal intelligence collection, analysis, and fusion.

**Integration with DSMIL**: DIRECTEYE tools interface directly with DSMIL devices via token-based API, providing external intelligence feeds.

### 6.2 DIRECTEYE Tool Categories

**Category 1: SIGINT (Signals Intelligence) – 8 tools**
- RF spectrum analysis
- Emitter identification
- Communications intercept
- Electronic warfare support

**Interfaces with**: Device 16 (SIGNALS, Layer 3)

**Category 2: IMINT (Imagery Intelligence) – 6 tools**
- Satellite imagery processing
- Aerial reconnaissance
- Change detection
- Object recognition

**Interfaces with**: Device 35 (Geospatial Intel, Layer 5), Device 41 (Treaty Monitoring, Layer 6)

**Category 3: HUMINT (Human Intelligence) – 4 tools**
- Source reporting
- Field intelligence
- Interrogation analysis
- Cultural intelligence

**Interfaces with**: Device 25 (Intel Fusion, Layer 4)

**Category 4: CYBER – 7 tools**
- Network traffic analysis
- Malware analysis
- APT tracking
- Vulnerability assessment

**Interfaces with**: Device 36 (Cyber Threat Prediction, Layer 5), Device 52 (Security AI, Layer 8)

**Category 5: OSINT (Open-Source Intelligence) – 5 tools**
- Web scraping
- Social media analysis
- News aggregation
- Entity extraction

**Interfaces with**: Device 49 (Global Intelligence OSINT, Layer 7)

**Category 6: GEOINT (Geospatial Intelligence) – 5 tools**
- GIS analysis
- Terrain modeling
- Infrastructure mapping
- Movement tracking

**Interfaces with**: Device 35 (Geospatial Intel, Layer 5)

### 6.3 DIRECTEYE Integration Architecture

```python
class DIRECTEYEIntegration:
    """
    Integration layer between DIRECTEYE tools and DSMIL devices.
    """

    TOOL_DEVICE_MAPPING = {
        # SIGINT tools → Device 16
        "rf_spectrum_analyzer": 16,
        "emitter_identifier": 16,
        "comms_intercept": 16,

        # IMINT tools → Device 35
        "satellite_processor": 35,
        "change_detector": 35,
        "object_recognizer": 35,

        # CYBER tools → Device 36, 52
        "network_analyzer": 36,
        "apt_tracker": 36,
        "malware_analyzer": 52,

        # OSINT tools → Device 49
        "web_scraper": 49,
        "social_analyzer": 49,
        "news_aggregator": 49,

        # Add all 35+ tools...
    }

    def send_tool_output_to_device(
        self,
        tool_name: str,
        tool_output: dict
    ) -> bool:
        """
        Send DIRECTEYE tool output to appropriate DSMIL device.
        """
        # Get target device
        device_id = self.TOOL_DEVICE_MAPPING.get(tool_name)
        if device_id is None:
            return False

        # Construct intelligence message
        token = 0x8000 + (device_id * 3) + 2  # DATA token
        message = {
            "source": "DIRECTEYE",
            "tool": tool_name,
            "data": tool_output,
            "timestamp": time.time(),
        }

        # Send to device
        return self.hil.send(token, message)

    def query_device_for_tool_input(
        self,
        tool_name: str,
        query_params: dict
    ) -> dict:
        """
        Query DSMIL device for input to DIRECTEYE tool.
        """
        # Reverse lookup: which device provides input for this tool?
        input_device_id = self.get_input_device_for_tool(tool_name)

        # Query device
        token = 0x8000 + (input_device_id * 3) + 2
        response = self.hil.send_and_receive(token, {
            "query": "TOOL_INPUT_REQUEST",
            "tool": tool_name,
            "params": query_params,
        })

        return response
```

### 6.4 Example: SIGINT Tool → Layer 3 Device 16 → Layer 7 Device 47

```text
┌─────────────────────────────────────────────────────────────────┐
│                 DIRECTEYE SIGINT → DSMIL Flow                   │
└─────────────────────────────────────────────────────────────────┘

1. DIRECTEYE RF Spectrum Analyzer
   ↓ Captures RF emissions, classifies signals
   ↓ Output: { "frequency": 1.2GHz, "emitter_type": "radar", "location": {...} }

2. DIRECTEYE Integration Layer
   ↓ Maps tool → Device 16 (SIGNALS, Layer 3)
   ↓ Sends via token 0x8032 (Device 16 DATA token)

3. Device 16 (SIGNALS, Layer 3)
   ↓ Model: "signal-classifier-int8" processes raw RF data
   ↓ Output: { "classification": "adversary_radar", "priority": "high" }
   ↓ Publishes event: "ADVERSARY_SIGNAL_DETECTED"

4. Device 25 (Intel Fusion, Layer 4) subscribes to "ADVERSARY_SIGNAL_DETECTED"
   ↓ Fuses with IMINT from Device 35
   ↓ Output: { "threat": "SAM site", "location": {...}, "confidence": 0.92 }

5. Device 47 (Advanced AI/ML, Layer 7)
   ↓ LLaMA-7B model synthesizes all intelligence
   ↓ Output: "High-priority SAM threat detected at coordinates X,Y. Recommend COA 1: Suppress. COA 2: Avoid. COA 3: Monitor."

6. Device 59 (Executive Command, Layer 9)
   ↓ Executive LLM provides final recommendation
   ↓ Output: "COA 1 (Suppress) recommended. Authorization required."
```

---

## 7. Event-Driven Intelligence

### 7.1 Event Bus Architecture

```python
class EventBus:
    """
    Pub-sub event bus for cross-layer intelligence flows.
    """

    def __init__(self):
        self.subscribers = {}  # {event_type: [(device_id, callback), ...]}

    def publish(self, event: dict) -> None:
        """
        Publish event to all subscribers.
        """
        event_type = event["event_type"]
        subscribers = self.subscribers.get(event_type, [])

        for device_id, callback in subscribers:
            # Clearance check
            if self.authorize_subscription(event["device_id"], device_id):
                callback(event)

    def subscribe(
        self,
        event_type: str,
        device_id: int,
        callback: callable
    ) -> None:
        """
        Subscribe device to event type.
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append((device_id, callback))

    def authorize_subscription(
        self,
        publisher_device_id: int,
        subscriber_device_id: int
    ) -> bool:
        """
        Authorize subscription (upward-only rule).
        """
        publisher_layer = router.DEVICE_LAYER_MAP[publisher_device_id]
        subscriber_layer = router.DEVICE_LAYER_MAP[subscriber_device_id]
        return subscriber_layer >= publisher_layer
```

### 7.2 Event Types

**Intelligence Events**:
- `SIGNAL_DETECTED` (Device 16 → Devices 25, 47)
- `THREAT_IDENTIFIED` (Device 25 → Devices 31, 47, 59)
- `PREDICTIVE_FORECAST` (Device 31 → Devices 47, 59)
- `STRATEGIC_ASSESSMENT` (Device 47 → Device 59)
- `EXECUTIVE_DECISION` (Device 59 → All layers for awareness)

**Security Events**:
- `INTRUSION_DETECTED` (Device 52 → Device 83, Device 59)
- `CLEARANCE_VIOLATION` (Any device → Device 52, Device 14)
- `DEEPFAKE_DETECTED` (Device 58 → Device 52, Device 59)

**System Events**:
- `MEMORY_THRESHOLD_EXCEEDED` (Any device → System Device 6)
- `DEVICE_OFFLINE` (HIL → Device 83, Device 59)
- `OPTIMIZATION_REQUIRED` (Any device → MLOps pipeline)

---

## 8. Workflow Examples

### 8.1 Example 1: Multi-INT Fusion → Strategic Assessment

**Scenario**: Adversary military buildup detected via multiple intelligence sources.

**Flow**:

```text
Step 1: SIGINT Detection (Layer 3)
   Device 16 (SIGNALS) detects increased radio traffic
   ↓ Event: "SIGNAL_ACTIVITY_INCREASED"

Step 2: IMINT Confirmation (Layer 5)
   Device 35 (Geospatial Intel) detects vehicle movements via satellite
   ↓ Event: "VEHICLE_MOVEMENT_DETECTED"

Step 3: HUMINT Correlation (Layer 4)
   Device 25 (Intel Fusion) receives field report via DIRECTEYE
   ↓ Fuses SIGINT + IMINT + HUMINT
   ↓ Event: "MILITARY_BUILDUP_CONFIRMED"

Step 4: Predictive Analysis (Layer 5)
   Device 31 (Predictive Analytics) forecasts timeline
   ↓ Output: "High probability of action within 48 hours"
   ↓ Event: "THREAT_TIMELINE_PREDICTED"

Step 5: Nuclear Assessment (Layer 6)
   Device 37 (ATOMAL Fusion) checks for nuclear dimensions
   ↓ Output: "No nuclear signature detected"

Step 6: Strategic Synthesis (Layer 7)
   Device 47 (Advanced AI/ML, LLaMA-7B) synthesizes all inputs
   ↓ Prompt: "Synthesize intelligence: SIGINT activity, IMINT movements, HUMINT reports, 48h timeline, no nuclear. Generate 3 COAs."
   ↓ Output:
      "COA 1: Preemptive diplomatic engagement
       COA 2: Forward-deploy assets to deter
       COA 3: Monitor and prepare response options"

Step 7: Security Validation (Layer 8)
   Device 52 (Security AI) validates intelligence chain
   ↓ No anomalies detected

Step 8: Executive Decision (Layer 9)
   Device 59 (Executive Command, Executive LLM) provides recommendation
   ↓ Input: All Layer 7 synthesis + strategic context
   ↓ Output: "Recommend COA 2 (Forward-deploy) with COA 1 (Diplomatic) in parallel. Authorize."
```

**Total Latency**: ~5 seconds (well within acceptable bounds for strategic decision)

**Memory Usage**:
- Layer 3: 0.6 GB (Device 16)
- Layer 4: 1.2 GB (Device 25)
- Layer 5: 3.4 GB (Devices 31 + 35)
- Layer 6: 2.2 GB (Device 37)
- Layer 7: 20.0 GB (Device 47)
- Layer 8: 1.0 GB (Device 52)
- Layer 9: 4.0 GB (Device 59)
- **Total**: 32.4 GB (within 62 GB budget)

### 8.2 Example 2: Cyber Threat → Emergency Response

**Scenario**: APT detected attempting to infiltrate Layer 7 (Advanced AI/ML).

**Flow**:

```text
Step 1: Intrusion Detection (Layer 8)
   Device 52 (Security AI) detects anomalous query pattern
   ↓ Classification: "APT-style lateral movement attempt"
   ↓ Event: "INTRUSION_DETECTED" (CRITICAL priority)

Step 2: Threat Analysis (Layer 5)
   Device 36 (Cyber Threat Prediction) analyzes attack vector
   ↓ Output: "Known APT28 TTPs, targeting Device 47 (LLM)"

Step 3: Immediate Response (Layer 8)
   Device 57 (Security Orchestration) triggers automated response
   ↓ Actions:
      - Isolate Device 47 network access
      - Capture forensic snapshot
      - Alert Layer 9

Step 4: Emergency Stop Evaluation (Device 83)
   Device 83 evaluates threat severity
   ↓ Decision: Partial halt (Device 47 only), not full system halt

Step 5: Executive Notification (Layer 9)
   Device 59 (Executive Command) receives alert
   ↓ Output: "Intrusion contained. Device 47 isolated. Forensics in progress."

Step 6: Post-Incident Analysis (Layer 7)
   Device 47 restored after forensic clearance
   ↓ Root cause: Exploited zero-day in query parser
   ↓ Remediation: Patch deployed via MLOps pipeline
```

**Total Latency**: ~200 ms (intrusion detection to containment)

---

## 9. Performance & Optimization

### 9.1 Latency Optimization

**Strategy 1: Event Coalescing**
- Batch multiple events from same source device
- Reduce cross-layer routing overhead by 40%

**Strategy 2: Predictive Prefetching**
- Layer 7 (Device 47) prefetches Layer 5–6 intelligence before explicit query
- Reduces latency by 60% for common workflows

**Strategy 3: Hot Path Caching**
- Cache frequent cross-layer queries (e.g., Device 47 → Device 16)
- 90% cache hit rate reduces latency from 500 ms → 50 ms

### 9.2 Bandwidth Optimization

**Total Cross-Layer Bandwidth Budget**: 64 GB/s (shared)

**Typical Bandwidth Usage**:
- Layer 3 → Layer 4: 2 GB/s (continuous domain analytics)
- Layer 4 → Layer 5: 1 GB/s (mission planning → predictive)
- Layer 5–6 → Layer 7: 4 GB/s (multi-source fusion)
- Layer 7 → Layer 9: 0.5 GB/s (strategic synthesis)
- Layer 8 ↔ All: 1 GB/s (security monitoring)
- **Total**: 8.5 GB/s (13% of bandwidth, well within budget)

**Optimization**: INT8 quantization reduces cross-layer data transfer by 4× (FP32 → INT8).

---

## 10. Implementation

### 10.1 Directory Structure

```text
/opt/dsmil/cross-layer/
├── routing/
│   ├── cross_layer_router.py       # Token-based routing
│   ├── security_gate.py            # Clearance enforcement
│   └── audit_logger.py             # Audit logging
├── orchestration/
│   ├── orchestrator.py             # 104-device orchestration
│   ├── pipeline_executor.py        # Sequential pipelines
│   └── parallel_executor.py        # Concurrent execution
├── events/
│   ├── event_bus.py                # Pub-sub event bus
│   ├── event_types.py              # Event type definitions
│   └── subscribers.py              # Device subscriptions
├── directeye/
│   ├── integration.py              # DIRECTEYE integration layer
│   ├── tool_mappings.py            # Tool → device mappings
│   └── tool_interfaces/            # Per-tool interfaces
│       ├── sigint_tools.py
│       ├── imint_tools.py
│       ├── cyber_tools.py
│       └── osint_tools.py
├── security/
│   ├── emergency_stop.py           # Device 83 emergency stop
│   ├── clearance_checker.py        # Clearance verification
│   └── forensics.py                # Forensic capture
└── monitoring/
    ├── flow_metrics.py             # Cross-layer flow metrics
    ├── latency_tracker.py          # Latency monitoring
    └── bandwidth_monitor.py        # Bandwidth usage
```

### 10.2 Configuration

```yaml
# /opt/dsmil/cross-layer/config.yaml

routing:
  upward_only_enforcement: true
  layer8_monitoring_required: true
  audit_all_cross_layer_queries: true

orchestration:
  max_concurrent_pipelines: 10
  pipeline_timeout_seconds: 60
  event_queue_size: 10000

directeye:
  enabled: true
  tool_count: 35
  default_timeout_seconds: 30

security:
  emergency_stop_device: 83
  layer8_security_devices: [51, 52, 53, 54, 55, 56, 57, 58]
  clearance_cache_ttl_seconds: 300

monitoring:
  latency_sampling_rate_hz: 10
  bandwidth_monitoring_enabled: true
  metrics_retention_days: 90
```

---

## Summary

This document defines **complete cross-layer intelligence flows** for the DSMIL 104-device architecture:

✅ **Upward-Only Flow**: Lower layers push to higher; downward queries blocked
✅ **Token-Based Routing**: 104 devices accessed via 0x8000-based tokens
✅ **Security Enforcement**: Clearance checks at every layer boundary
✅ **Event-Driven Architecture**: Pub-sub model for asynchronous intelligence flow
✅ **DIRECTEYE Integration**: 35+ tools interface with DSMIL devices
✅ **Orchestration Modes**: Pipeline, parallel, event-driven execution
✅ **Emergency Stop**: Device 83 hardware-enforced system halt
✅ **Audit Logging**: Comprehensive audit trail for all cross-layer queries

**Key Insights**:

1. **Layer 7 (Device 47) is the synthesis hub**: Receives intelligence from Layers 2–6, provides strategic reasoning
2. **Layer 8 provides security overlay**: Monitors all cross-layer flows, triggers Device 83 on breach
3. **DIRECTEYE extends intelligence collection**: 35+ tools feed DSMIL devices with multi-INT data
4. **Event-driven reduces latency**: Pub-sub eliminates blocking cross-layer queries
5. **Bandwidth is optimized**: 8.5 GB/s typical usage (13% of 64 GB/s budget)

**Next Document**:
- **07_IMPLEMENTATION_ROADMAP.md**: 6-phase implementation plan with milestones, resource requirements, and success criteria

---

**End of Cross-Layer Intelligence Flows & Orchestration (Version 1.0)**
