# Phase 10 – Exercise & Simulation Framework (v1.0)

**Version:** 1.0
**Status:** Initial Release
**Date:** 2025-11-23
**Prerequisite:** Phase 9 (Operations & Incident Response)
**Next Phase:** Phase 11 (External Military Communications Integration)

---

## 1. Objectives

Phase 10 establishes a comprehensive **Exercise & Simulation Framework** enabling:

1. **Multi-tenant exercise management** with EXERCISE_ALPHA, EXERCISE_BRAVO, ATOMAL_EXERCISE
2. **Synthetic event injection** for L3-L9 training across all intelligence types
3. **Red team simulation engine** with adaptive adversary tactics
4. **After-action reporting** with SHRINK stress analysis and decision tree visualization
5. **Exercise data segregation** from operational production data

### System Context (v3.1)

- **Physical Hardware:** Intel Core Ultra 7 165H (48.2 TOPS INT8: 13.0 NPU + 32.0 GPU + 3.2 CPU)
- **Memory:** 64 GB LPDDR5x-7467, 62 GB usable for AI, 64 GB/s shared bandwidth
- **Phase 10 Allocation:** 10 devices (63-72), 2 GB budget, 4.0 TOPS (GPU-primary)
  - Device 63: Exercise Controller (200 MB, orchestration)
  - Device 64: Scenario Engine (250 MB, JSON scenario processing)
  - Device 65-67: Synthetic Event Injectors (150 MB each, SIGINT/IMINT/HUMINT)
  - Device 68: Red Team Simulation (400 MB, adversary modeling)
  - Device 69: Blue Force Tracking (200 MB, friendly unit simulation)
  - Device 70: After-Action Report Generator (300 MB, metrics + visualization)
  - Device 71: Training Assessment System (200 MB, performance scoring)
  - Device 72: Exercise Data Recorder (300 MB, full message capture)

### Key Principles

1. **Exercise data MUST be segregated** from operational data (separate Redis/Postgres schemas)
2. **ROE_LEVEL=TRAINING required** during all exercises (enforced at protocol level)
3. **ATOMAL exercises require two-person authorization** (dual ML-DSA-87 signatures)
4. **No kinetic outputs during TRAINING mode** (Device 61 NC3 Integration disabled)
5. **Realistic adversary simulation** with adaptive tactics and false positives

---

## 2. Architecture Overview

### 2.1 Phase 10 Service Topology

```
┌───────────────────────────────────────────────────────────────┐
│              Phase 10 - Exercise Framework                     │
│           Devices 63-72, 2 GB Budget, 4.0 TOPS                │
└───────────────────────────────────────────────────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
 ┌────▼──────┐      ┌────────▼────────┐     ┌──────▼──────┐
 │ Exercise  │      │ Scenario Engine │     │ Red Team    │
 │Controller │◄─────┤   (Device 64)   │────►│Simulation   │
 │(Device 63)│ DBE  │ JSON Scenarios  │ DBE │ (Device 68) │
 └────┬──────┘      └─────────────────┘     └──────┬──────┘
      │                     │                       │
      │ Exercise Control    │ Event Injection       │ Attack Injection
      │ TLVs (0x90-0x9F)    │ TLVs (0x93)          │ TLVs (0x94)
      │                     │                       │
 ┌────▼─────────────────────▼───────────────────────▼──────┐
 │              L3 Ingestion Layer (Devices 14-16)          │
 │   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
 │   │ SIGINT  │   │ IMINT   │   │ HUMINT  │              │
 │   │ Inject  │   │ Inject  │   │ Inject  │              │
 │   │(Dev 65) │   │(Dev 66) │   │(Dev 67) │              │
 │   └─────────┘   └─────────┘   └─────────┘              │
 └──────────────────────────────────────────────────────────┘
                      │
                      │ Real-time event flow
                      │ during exercise
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │         L3-L9 Processing Pipeline (Training Mode)         │
 │    L3 (Adaptive) → L4 (Reactive) → L5 (Predictive) →     │
 │    L6 (Proactive) → L7 (Extended AI) → L8 (Enhanced) →   │
 │    L9 (Executive - TRAINING only)                         │
 └──────────────────────────────────────────────────────────┘
                      │
                      │ All events recorded
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │           Exercise Data Recorder (Device 72)              │
 │     Full DBE capture + replay + after-action review       │
 └──────────────────────────────────────────────────────────┘
                      │
                      │ Post-exercise analysis
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │      After-Action Report Generator (Device 70)            │
 │   Metrics, decision trees, SHRINK analysis, timeline      │
 └──────────────────────────────────────────────────────────┘
```

### 2.2 Phase 10 Services

| Service | Device | Token IDs | Memory | Purpose |
|---------|--------|-----------|--------|---------|
| `dsmil-exercise-controller` | 63 | 0x80BD-0x80BF | 200 MB | Exercise lifecycle management |
| `dsmil-scenario-engine` | 64 | 0x80C0-0x80C2 | 250 MB | JSON scenario processing |
| `dsmil-sigint-injector` | 65 | 0x80C3-0x80C5 | 150 MB | SIGINT event synthesis |
| `dsmil-imint-injector` | 66 | 0x80C6-0x80C8 | 150 MB | IMINT event synthesis |
| `dsmil-humint-injector` | 67 | 0x80C9-0x80CB | 150 MB | HUMINT event synthesis |
| `dsmil-redteam-engine` | 68 | 0x80CC-0x80CE | 400 MB | Adversary behavior modeling |
| `dsmil-blueforce-sim` | 69 | 0x80CF-0x80D1 | 200 MB | Friendly unit tracking |
| `dsmil-aar-generator` | 70 | 0x80D2-0x80D4 | 300 MB | After-action report generation |
| `dsmil-training-assess` | 71 | 0x80D5-0x80D7 | 200 MB | Performance scoring |
| `dsmil-exercise-recorder` | 72 | 0x80D8-0x80DA | 300 MB | Full message capture |

### 2.3 DBE Message Types for Phase 10

**New `msg_type` definitions (Exercise Control 0x90-0x9F):**

| Message Type | Hex | Purpose | Direction |
|--------------|-----|---------|-----------|
| `EXERCISE_START` | `0x90` | Initiate exercise with tenant config | Controller → All |
| `EXERCISE_STOP` | `0x91` | Terminate exercise and begin AAR | Controller → All |
| `EXERCISE_PAUSE` | `0x92` | Pause event injection (white cell break) | Controller → Injectors |
| `INJECT_EVENT` | `0x93` | Synthetic event injection command | Scenario → Injectors |
| `RED_TEAM_ACTION` | `0x94` | Adversary action injection | RedTeam → L3 |
| `SCENARIO_CHECKPOINT` | `0x95` | Scenario milestone reached | Scenario → Controller |
| `EXERCISE_STATUS` | `0x96` | Current exercise state query | Any → Controller |
| `AAR_REQUEST` | `0x97` | Request after-action report | Controller → AAR Gen |
| `TRAINING_METRIC` | `0x98` | Performance metric update | Assess → Controller |

**DBE Header TLVs for Phase 10 (extended from Phase 7 spec):**

```text
EXERCISE_TENANT_ID (string)     – e.g., "EXERCISE_ALPHA", "ATOMAL_EXERCISE"
SCENARIO_ID (UUID)               – Unique scenario identifier
EXERCISE_TIMESTAMP (uint64)      – Exercise time (may differ from real time)
INJECT_TYPE (enum)               – SIGINT, IMINT, HUMINT, CYBER, PHYSICAL
EVENT_REALISM (float)            – 0.0-1.0 (noise/false positive rate)
RED_TEAM_UNIT (string)           – Simulated adversary unit ID
BLUE_FORCE_UNIT (string)         – Simulated friendly unit ID
EXERCISE_PHASE (enum)            – SETUP, EXECUTION, WHITE_CELL, AAR
DUAL_AUTH_SIG_1 (blob)           – First ML-DSA-87 signature (ATOMAL exercises)
DUAL_AUTH_SIG_2 (blob)           – Second ML-DSA-87 signature (ATOMAL exercises)
```

---

## 3. Device 63: Exercise Controller

**Purpose:** Central orchestrator for all exercise lifecycle management.

**Token IDs:**
- `0x80BD` (STATUS): Current exercise state, active tenant, scenario progress
- `0x80BE` (CONFIG): Exercise configuration, tenant definitions, authorization
- `0x80BF` (DATA): Exercise metadata, participant roster, objectives

**Responsibilities:**

1. **Tenant Management:**
   - Create exercise tenants: EXERCISE_ALPHA (SECRET), EXERCISE_BRAVO (TOP_SECRET), ATOMAL_EXERCISE (ATOMAL)
   - Enforce tenant isolation in Redis/Postgres
   - Track participant access per tenant

2. **Exercise Lifecycle:**
   - **SETUP:** Load scenario, configure injectors, verify participant auth
   - **EXECUTION:** Monitor event injection, track objectives, enforce ROE_LEVEL=TRAINING
   - **WHITE_CELL:** Pause for observer intervention or scenario adjustment
   - **AAR:** Trigger data collection, generate reports, archive exercise data

3. **Authorization:**
   - ATOMAL exercises require two-person authorization (dual ML-DSA-87 signatures)
   - Validate `DUAL_AUTH_SIG_1` and `DUAL_AUTH_SIG_2` against authorized exercise directors
   - Enforce need-to-know for ATOMAL exercise data access

4. **ROE Enforcement:**
   - Set global `ROE_LEVEL=TRAINING` for all L3-L9 devices during exercise
   - Disable Device 61 (NC3 Integration) to prevent kinetic outputs
   - Restore operational ROE levels after exercise completion

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/exercise_controller.py
"""
DSMIL Exercise Controller (Device 63)
Central orchestrator for exercise lifecycle management
"""

import time
import logging
import redis
import psycopg2
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from dsmil_dbe import DBEMessage, DBESocket, MessageType
from dsmil_pqc import MLDSAVerifier

# Constants
DEVICE_ID = 63
TOKEN_BASE = 0x80BD
REDIS_HOST = "localhost"
POSTGRES_HOST = "localhost"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [EXERCISE-CTRL] [Device-63] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class ExercisePhase(Enum):
    IDLE = 0
    SETUP = 1
    EXECUTION = 2
    WHITE_CELL = 3
    AAR = 4

class TenantType(Enum):
    EXERCISE_ALPHA = "SECRET"
    EXERCISE_BRAVO = "TOP_SECRET"
    ATOMAL_EXERCISE = "ATOMAL"

@dataclass
class ExerciseTenant:
    tenant_id: str
    classification: str
    scenario_id: str
    start_time: float
    participants: List[str]
    dual_auth_required: bool
    auth_signature_1: Optional[bytes] = None
    auth_signature_2: Optional[bytes] = None

class ExerciseController:
    def __init__(self):
        self.current_phase = ExercisePhase.IDLE
        self.active_tenant: Optional[ExerciseTenant] = None

        # Connect to Redis (exercise-specific schemas)
        self.redis = redis.Redis(host=REDIS_HOST, db=15)  # DB 15 for exercises

        # Connect to Postgres (exercise-specific database)
        self.pg = psycopg2.connect(
            host=POSTGRES_HOST,
            database="exercise_db",
            user="dsmil_exercise",
            password="<from-vault>"
        )

        # DBE socket for receiving control messages
        self.dbe_socket = DBESocket("/var/run/dsmil/exercise-controller.sock")

        # PQC verifier for dual authorization
        self.verifier = MLDSAVerifier()

        logger.info(f"Exercise Controller initialized (Device {DEVICE_ID})")

    def start_exercise(self, request: DBEMessage) -> DBEMessage:
        """
        Start a new exercise session

        Required TLVs:
        - EXERCISE_TENANT_ID
        - SCENARIO_ID
        - CLASSIFICATION
        - DUAL_AUTH_SIG_1 (if ATOMAL)
        - DUAL_AUTH_SIG_2 (if ATOMAL)
        """
        tenant_id = request.tlv_get("EXERCISE_TENANT_ID")
        scenario_id = request.tlv_get("SCENARIO_ID")
        classification = request.tlv_get("CLASSIFICATION")

        # Validate not already running
        if self.current_phase != ExercisePhase.IDLE:
            return self._error_response("EXERCISE_ALREADY_ACTIVE",
                                       f"Current phase: {self.current_phase.name}")

        # Check dual authorization for ATOMAL
        dual_auth_required = (classification == "ATOMAL")
        if dual_auth_required:
            sig1 = request.tlv_get("DUAL_AUTH_SIG_1")
            sig2 = request.tlv_get("DUAL_AUTH_SIG_2")

            if not sig1 or not sig2:
                return self._error_response("MISSING_DUAL_AUTH",
                                           "ATOMAL exercises require two signatures")

            # Verify signatures
            auth_message = f"{tenant_id}:{scenario_id}:{classification}:{time.time()}"
            if not self.verifier.verify(auth_message.encode(), sig1):
                return self._error_response("INVALID_AUTH_SIG_1", "First signature invalid")
            if not self.verifier.verify(auth_message.encode(), sig2):
                return self._error_response("INVALID_AUTH_SIG_2", "Second signature invalid")

            # Verify different signers (public keys must differ)
            if self.verifier.get_pubkey(sig1) == self.verifier.get_pubkey(sig2):
                return self._error_response("SAME_SIGNER", "Signatures must be from different authorized personnel")

        # Create tenant
        self.active_tenant = ExerciseTenant(
            tenant_id=tenant_id,
            classification=classification,
            scenario_id=scenario_id,
            start_time=time.time(),
            participants=[],
            dual_auth_required=dual_auth_required,
            auth_signature_1=request.tlv_get("DUAL_AUTH_SIG_1") if dual_auth_required else None,
            auth_signature_2=request.tlv_get("DUAL_AUTH_SIG_2") if dual_auth_required else None
        )

        # Initialize Redis schema
        self.redis.flushdb()  # Clear previous exercise data
        self.redis.set(f"exercise:{tenant_id}:status", "SETUP")
        self.redis.set(f"exercise:{tenant_id}:scenario_id", scenario_id)
        self.redis.set(f"exercise:{tenant_id}:classification", classification)

        # Initialize Postgres tables
        with self.pg.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {tenant_id}_events (
                    event_id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    device_id INT NOT NULL,
                    payload JSONB NOT NULL
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {tenant_id}_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    device_id INT NOT NULL
                )
            """)
            self.pg.commit()

        # Set global ROE_LEVEL=TRAINING for all L3-L9 devices
        self._set_global_roe("TRAINING")

        # Disable Device 61 (NC3 Integration) to prevent kinetic outputs
        self._disable_nc3()

        # Transition to SETUP phase
        self.current_phase = ExercisePhase.SETUP

        logger.info(f"Exercise started: {tenant_id}, Scenario: {scenario_id}, "
                   f"Classification: {classification}, Dual-Auth: {dual_auth_required}")

        # Notify all Phase 10 devices
        self._broadcast_exercise_start()

        return self._success_response("EXERCISE_STARTED", {
            "tenant_id": tenant_id,
            "scenario_id": scenario_id,
            "phase": "SETUP"
        })

    def stop_exercise(self, request: DBEMessage) -> DBEMessage:
        """
        Stop current exercise and initiate AAR
        """
        if self.current_phase == ExercisePhase.IDLE:
            return self._error_response("NO_ACTIVE_EXERCISE", "Cannot stop - no exercise running")

        if not self.active_tenant:
            return self._error_response("INVALID_STATE", "Active tenant is None")

        tenant_id = self.active_tenant.tenant_id

        # Transition to AAR phase
        self.current_phase = ExercisePhase.AAR
        self.redis.set(f"exercise:{tenant_id}:status", "AAR")

        # Stop event injection
        self._broadcast_exercise_stop()

        # Trigger AAR generation (Device 70)
        self._request_aar_generation()

        # Restore operational ROE levels
        self._restore_operational_roe()

        # Re-enable Device 61 (NC3 Integration)
        self._enable_nc3()

        logger.info(f"Exercise stopped: {tenant_id}, entering AAR phase")

        return self._success_response("EXERCISE_STOPPED", {
            "tenant_id": tenant_id,
            "phase": "AAR"
        })

    def _set_global_roe(self, roe_level: str):
        """Set ROE_LEVEL for all L3-L9 devices"""
        for device_id in range(14, 63):  # Devices 14-62 (L3-L9)
            token_config = 0x8000 + (device_id * 3) + 1  # CONFIG token
            self.redis.set(f"device:{device_id}:roe_level", roe_level)
            logger.debug(f"Set Device {device_id} ROE_LEVEL={roe_level}")

    def _disable_nc3(self):
        """Disable Device 61 (NC3 Integration) during exercises"""
        self.redis.set("device:61:enabled", "false")
        logger.warning("Device 61 (NC3 Integration) DISABLED for exercise safety")

    def _enable_nc3(self):
        """Re-enable Device 61 (NC3 Integration) after exercise"""
        self.redis.set("device:61:enabled", "true")
        logger.info("Device 61 (NC3 Integration) RE-ENABLED post-exercise")

    def _restore_operational_roe(self):
        """Restore pre-exercise ROE levels"""
        # Default operational ROE is ANALYSIS_ONLY for most devices
        self._set_global_roe("ANALYSIS_ONLY")
        logger.info("Operational ROE levels restored")

    def _broadcast_exercise_start(self):
        """Notify all Phase 10 devices of exercise start"""
        msg = DBEMessage(
            msg_type=0x90,  # EXERCISE_START
            device_id_src=DEVICE_ID,
            device_id_dst=0xFF,  # Broadcast
            tlvs={
                "EXERCISE_TENANT_ID": self.active_tenant.tenant_id,
                "SCENARIO_ID": self.active_tenant.scenario_id,
                "CLASSIFICATION": self.active_tenant.classification,
                "EXERCISE_PHASE": "EXECUTION"
            }
        )

        # Send to Scenario Engine (Device 64)
        self.dbe_socket.send_to("/var/run/dsmil/scenario-engine.sock", msg)

        # Send to Event Injectors (Devices 65-67)
        for device_id in range(65, 68):
            sock_path = f"/var/run/dsmil/event-injector-{device_id}.sock"
            self.dbe_socket.send_to(sock_path, msg)

        # Send to Red Team Engine (Device 68)
        self.dbe_socket.send_to("/var/run/dsmil/redteam-engine.sock", msg)

        # Send to Exercise Recorder (Device 72)
        self.dbe_socket.send_to("/var/run/dsmil/exercise-recorder.sock", msg)

        logger.info("Broadcast EXERCISE_START to all Phase 10 devices")

    def _broadcast_exercise_stop(self):
        """Notify all Phase 10 devices of exercise stop"""
        msg = DBEMessage(
            msg_type=0x91,  # EXERCISE_STOP
            device_id_src=DEVICE_ID,
            device_id_dst=0xFF,  # Broadcast
            tlvs={
                "EXERCISE_TENANT_ID": self.active_tenant.tenant_id,
                "EXERCISE_PHASE": "AAR"
            }
        )

        # Broadcast to all Phase 10 devices
        for device_id in range(64, 73):
            sock_path = f"/var/run/dsmil/device-{device_id}.sock"
            try:
                self.dbe_socket.send_to(sock_path, msg)
            except Exception as e:
                logger.warning(f"Failed to notify Device {device_id}: {e}")

        logger.info("Broadcast EXERCISE_STOP to all Phase 10 devices")

    def _request_aar_generation(self):
        """Request After-Action Report from Device 70"""
        msg = DBEMessage(
            msg_type=0x97,  # AAR_REQUEST
            device_id_src=DEVICE_ID,
            device_id_dst=70,
            tlvs={
                "EXERCISE_TENANT_ID": self.active_tenant.tenant_id,
                "SCENARIO_ID": self.active_tenant.scenario_id,
                "START_TIME": str(self.active_tenant.start_time),
                "END_TIME": str(time.time())
            }
        )

        self.dbe_socket.send_to("/var/run/dsmil/aar-generator.sock", msg)
        logger.info("Requested AAR generation from Device 70")

    def _success_response(self, status: str, data: Dict) -> DBEMessage:
        """Build success response"""
        return DBEMessage(
            msg_type=0x96,  # EXERCISE_STATUS
            device_id_src=DEVICE_ID,
            tlvs={
                "STATUS": status,
                "DATA": str(data)
            }
        )

    def _error_response(self, error_code: str, error_msg: str) -> DBEMessage:
        """Build error response"""
        logger.error(f"Error: {error_code} - {error_msg}")
        return DBEMessage(
            msg_type=0x96,  # EXERCISE_STATUS
            device_id_src=DEVICE_ID,
            tlvs={
                "STATUS": "ERROR",
                "ERROR_CODE": error_code,
                "ERROR_MSG": error_msg
            }
        )

    def run(self):
        """Main event loop"""
        logger.info("Exercise Controller running, waiting for commands...")

        while True:
            try:
                msg = self.dbe_socket.receive()

                if msg.msg_type == 0x90:  # EXERCISE_START
                    response = self.start_exercise(msg)
                    self.dbe_socket.send(response)

                elif msg.msg_type == 0x91:  # EXERCISE_STOP
                    response = self.stop_exercise(msg)
                    self.dbe_socket.send(response)

                elif msg.msg_type == 0x96:  # EXERCISE_STATUS query
                    response = self._get_status()
                    self.dbe_socket.send(response)

                else:
                    logger.warning(f"Unknown message type: 0x{msg.msg_type:02X}")

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)

    def _get_status(self) -> DBEMessage:
        """Return current exercise status"""
        if self.active_tenant:
            return self._success_response("ACTIVE", {
                "phase": self.current_phase.name,
                "tenant_id": self.active_tenant.tenant_id,
                "scenario_id": self.active_tenant.scenario_id,
                "classification": self.active_tenant.classification,
                "uptime_seconds": time.time() - self.active_tenant.start_time
            })
        else:
            return self._success_response("IDLE", {"phase": "IDLE"})

if __name__ == "__main__":
    controller = ExerciseController()
    controller.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-exercise-controller.service
[Unit]
Description=DSMIL Exercise Controller (Device 63)
After=network.target redis.service postgresql.service
Requires=redis.service postgresql.service

[Service]
Type=simple
User=dsmil
Group=dsmil
ExecStart=/usr/bin/python3 /opt/dsmil/exercise_controller.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
PrivateTmp=yes
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/run/dsmil /var/log/dsmil

[Install]
WantedBy=multi-user.target
```

---

## 4. Device 64: Scenario Engine

**Purpose:** Load and execute JSON-based exercise scenarios with timeline control.

**Token IDs:**
- `0x80C0` (STATUS): Current scenario state, active checkpoint, progress %
- `0x80C1` (CONFIG): Scenario file path, execution parameters
- `0x80C2` (DATA): Scenario JSON content, event queue

**Scenario JSON Format:**

```json
{
  "scenario_id": "cyber-apt-attack-2025",
  "name": "APT Cyber Attack Simulation",
  "classification": "SECRET",
  "duration_minutes": 240,
  "objectives": [
    "Detect initial reconnaissance within 30 minutes",
    "Identify C2 infrastructure within 2 hours",
    "Contain lateral movement before data exfiltration"
  ],
  "timeline": [
    {
      "time_offset_minutes": 0,
      "event_type": "INJECT_EVENT",
      "target_device": 65,
      "inject_type": "SIGINT",
      "payload": {
        "intercept_type": "network_scan",
        "source_ip": "203.0.113.45",
        "target_ip": "10.0.1.0/24",
        "ports": [22, 23, 80, 443, 8080],
        "timestamp": "2025-11-23T14:00:00Z"
      }
    },
    {
      "time_offset_minutes": 15,
      "event_type": "RED_TEAM_ACTION",
      "target_device": 68,
      "action": "phishing_email",
      "payload": {
        "target_user": "john.doe@example.mil",
        "subject": "Urgent: Security Update Required",
        "malicious_link": "http://203.0.113.45/update.exe",
        "success_probability": 0.3
      }
    },
    {
      "time_offset_minutes": 45,
      "event_type": "SCENARIO_CHECKPOINT",
      "checkpoint_name": "Initial Access Achieved",
      "success_criteria": {
        "l3_alert_triggered": true,
        "l4_incident_created": true
      }
    }
  ],
  "red_team_units": [
    {
      "unit_id": "APT-EMULATOR-1",
      "tactics": ["reconnaissance", "initial_access", "persistence"],
      "sophistication": 0.8
    }
  ],
  "blue_force_units": [
    {
      "unit_id": "SOC-TEAM-ALPHA",
      "location": "CONUS",
      "shift_schedule": "24/7"
    }
  ]
}
```

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/scenario_engine.py
"""
DSMIL Scenario Engine (Device 64)
Loads and executes JSON exercise scenarios
"""

import json
import time
import threading
import logging
from typing import Dict, List
from dataclasses import dataclass

from dsmil_dbe import DBEMessage, DBESocket, MessageType

DEVICE_ID = 64
TOKEN_BASE = 0x80C0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SCENARIO-ENGINE] [Device-64] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScenarioEvent:
    time_offset_minutes: int
    event_type: str
    target_device: int
    payload: Dict

class ScenarioEngine:
    def __init__(self):
        self.current_scenario: Optional[Dict] = None
        self.scenario_start_time: Optional[float] = None
        self.event_queue: List[ScenarioEvent] = []
        self.execution_thread: Optional[threading.Thread] = None
        self.running = False

        self.dbe_socket = DBESocket("/var/run/dsmil/scenario-engine.sock")

        logger.info(f"Scenario Engine initialized (Device {DEVICE_ID})")

    def load_scenario(self, scenario_path: str):
        """Load scenario from JSON file"""
        try:
            with open(scenario_path, 'r') as f:
                self.current_scenario = json.load(f)

            # Validate required fields
            required = ["scenario_id", "name", "classification", "timeline"]
            for field in required:
                if field not in self.current_scenario:
                    raise ValueError(f"Missing required field: {field}")

            # Parse timeline into event queue
            self.event_queue = []
            for event_data in self.current_scenario["timeline"]:
                event = ScenarioEvent(
                    time_offset_minutes=event_data["time_offset_minutes"],
                    event_type=event_data["event_type"],
                    target_device=event_data.get("target_device", 0),
                    payload=event_data.get("payload", {})
                )
                self.event_queue.append(event)

            # Sort by time offset
            self.event_queue.sort(key=lambda e: e.time_offset_minutes)

            logger.info(f"Loaded scenario: {self.current_scenario['name']}, "
                       f"{len(self.event_queue)} events")

        except Exception as e:
            logger.error(f"Failed to load scenario: {e}", exc_info=True)
            raise

    def start_execution(self):
        """Start scenario execution"""
        if not self.current_scenario:
            raise ValueError("No scenario loaded")

        if self.running:
            raise ValueError("Scenario already running")

        self.scenario_start_time = time.time()
        self.running = True

        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()

        logger.info(f"Started scenario execution: {self.current_scenario['scenario_id']}")

    def stop_execution(self):
        """Stop scenario execution"""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5)

        logger.info("Stopped scenario execution")

    def _execution_loop(self):
        """Main execution loop - inject events at scheduled times"""
        event_index = 0

        while self.running and event_index < len(self.event_queue):
            event = self.event_queue[event_index]

            # Calculate target time
            target_time = self.scenario_start_time + (event.time_offset_minutes * 60)

            # Wait until target time
            while time.time() < target_time and self.running:
                time.sleep(1)

            if not self.running:
                break

            # Execute event
            try:
                self._execute_event(event)
                event_index += 1
            except Exception as e:
                logger.error(f"Failed to execute event {event_index}: {e}", exc_info=True)
                # Continue with next event
                event_index += 1

        logger.info("Scenario execution completed")
        self.running = False

    def _execute_event(self, event: ScenarioEvent):
        """Execute a single scenario event"""
        logger.info(f"Executing event: {event.event_type} → Device {event.target_device}")

        if event.event_type == "INJECT_EVENT":
            # Send to Event Injector (Devices 65-67)
            msg = DBEMessage(
                msg_type=0x93,  # INJECT_EVENT
                device_id_src=DEVICE_ID,
                device_id_dst=event.target_device,
                tlvs={
                    "INJECT_TYPE": event.payload.get("inject_type", "SIGINT"),
                    "PAYLOAD": json.dumps(event.payload),
                    "SCENARIO_ID": self.current_scenario["scenario_id"]
                }
            )
            target_sock = f"/var/run/dsmil/event-injector-{event.target_device}.sock"
            self.dbe_socket.send_to(target_sock, msg)

        elif event.event_type == "RED_TEAM_ACTION":
            # Send to Red Team Engine (Device 68)
            msg = DBEMessage(
                msg_type=0x94,  # RED_TEAM_ACTION
                device_id_src=DEVICE_ID,
                device_id_dst=68,
                tlvs={
                    "ACTION": event.payload.get("action", "unknown"),
                    "PAYLOAD": json.dumps(event.payload),
                    "SCENARIO_ID": self.current_scenario["scenario_id"]
                }
            )
            self.dbe_socket.send_to("/var/run/dsmil/redteam-engine.sock", msg)

        elif event.event_type == "SCENARIO_CHECKPOINT":
            # Send checkpoint notification to Exercise Controller (Device 63)
            msg = DBEMessage(
                msg_type=0x95,  # SCENARIO_CHECKPOINT
                device_id_src=DEVICE_ID,
                device_id_dst=63,
                tlvs={
                    "CHECKPOINT_NAME": event.payload.get("checkpoint_name", "Unnamed"),
                    "SUCCESS_CRITERIA": json.dumps(event.payload.get("success_criteria", {})),
                    "SCENARIO_ID": self.current_scenario["scenario_id"]
                }
            )
            self.dbe_socket.send_to("/var/run/dsmil/exercise-controller.sock", msg)

        else:
            logger.warning(f"Unknown event type: {event.event_type}")

if __name__ == "__main__":
    engine = ScenarioEngine()
    # Wait for EXERCISE_START message from Controller
    logger.info("Waiting for exercise start...")
```

---

## 5. Devices 65-67: Synthetic Event Injectors

**Purpose:** Generate realistic SIGINT, IMINT, HUMINT events for L3 ingestion during exercises.

### Device 65: SIGINT Event Injector (0x80C3-0x80C5)

**Capabilities:**
- Network intercepts (TCP/UDP packet captures)
- ELINT (electronic intelligence - radar emissions, jamming)
- COMINT (communications intelligence - radio intercepts, phone calls)
- Cyber indicators (malware signatures, C2 beacons)

**Realism Features:**
- Noise injection (false positives, decoy traffic)
- Timing jitter (realistic network delays)
- Incomplete data (partial intercepts, corruption)

**Implementation Sketch:**

```python
#!/usr/bin/env python3
# /opt/dsmil/sigint_injector.py
"""
DSMIL SIGINT Event Injector (Device 65)
Generates synthetic SIGINT events for exercises
"""

import time
import random
import logging
from typing import Dict

from dsmil_dbe import DBEMessage, DBESocket

DEVICE_ID = 65
TOKEN_BASE = 0x80C3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SIGINTInjector:
    def __init__(self):
        self.dbe_socket = DBESocket("/var/run/dsmil/event-injector-65.sock")
        self.l3_sigint_device = 14  # Device 14: SIGINT ingestion

        logger.info(f"SIGINT Injector initialized (Device {DEVICE_ID})")

    def inject_network_scan(self, payload: Dict):
        """Inject simulated network reconnaissance"""
        # Add realism: noise, timing jitter
        realism = payload.get("realism", 0.9)

        # Generate scan data
        scan_data = {
            "source_ip": payload["source_ip"],
            "target_ip": payload["target_ip"],
            "ports": payload["ports"],
            "timestamp": time.time(),
            "confidence": realism,
            "sensor_id": "SIGINT-SENSOR-03"
        }

        # Add false positives based on realism
        if random.random() > realism:
            scan_data["false_positive"] = True
            scan_data["noise_reason"] = "network_congestion"

        # Send to L3 SIGINT ingestion (Device 14)
        msg = DBEMessage(
            msg_type=0x21,  # L3_INGEST (from Phase 3 spec)
            device_id_src=DEVICE_ID,
            device_id_dst=self.l3_sigint_device,
            tlvs={
                "INJECT_TYPE": "SIGINT",
                "EVENT_TYPE": "network_scan",
                "PAYLOAD": str(scan_data),
                "CLASSIFICATION": "SECRET",
                "EXERCISE_TENANT_ID": payload.get("tenant_id", "EXERCISE_ALPHA")
            }
        )

        self.dbe_socket.send_to("/var/run/dsmil/l3-sigint.sock", msg)
        logger.info(f"Injected network scan: {scan_data['source_ip']} → {scan_data['target_ip']}")
```

### Device 66: IMINT Event Injector (0x80C6-0x80C8)

**Capabilities:**
- Satellite imagery (SAR, optical, thermal)
- Drone/UAV footage
- Reconnaissance photos
- Geospatial intelligence (GEOINT)

**Realism Features:**
- Cloud cover (obscured targets)
- Resolution limits (pixelated, low-quality)
- Timestamp delays (satellite revisit times)

### Device 67: HUMINT Event Injector (0x80C9-0x80CB)

**Capabilities:**
- Agent reports (field operatives)
- Interrogation transcripts
- Source debriefs
- Walk-in volunteers

**Realism Features:**
- Credibility scoring (unreliable sources)
- Translation errors (foreign language reports)
- Delayed reporting (agent safety)

---

## 6. Device 68: Red Team Simulation Engine

**Purpose:** Model adversary behavior with adaptive tactics.

**Token IDs:**
- `0x80CC` (STATUS): Current attack phase, success rate, detection status
- `0x80CD` (CONFIG): Adversary profile, sophistication level, objectives
- `0x80CE` (DATA): Attack timeline, TTP (Tactics, Techniques, Procedures)

**Adversary Behavior Models:**

| Model | Description | Tactics | Sophistication |
|-------|-------------|---------|----------------|
| APT-Style | Advanced Persistent Threat | Stealth, persistence, exfiltration | 0.8-1.0 |
| Insider-Threat | Malicious insider | Privilege abuse, data theft | 0.5-0.7 |
| Ransomware | Financially-motivated | Encryption, extortion | 0.4-0.6 |
| Script-Kiddie | Low-skill attacker | Automated tools, public exploits | 0.1-0.3 |

**Adaptive Tactics:**
- If blue team detects recon, switch to low-and-slow approach
- If firewall blocks C2, switch to DNS tunneling
- If EDR deployed, use fileless malware
- If network segmented, pivot to VPN access

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/redteam_engine.py
"""
DSMIL Red Team Simulation Engine (Device 68)
Models adversary behavior with adaptive tactics
"""

import time
import random
import logging
from typing import Dict, List
from enum import Enum

from dsmil_dbe import DBEMessage, DBESocket

DEVICE_ID = 68
TOKEN_BASE = 0x80CC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttackPhase(Enum):
    RECONNAISSANCE = 1
    INITIAL_ACCESS = 2
    PERSISTENCE = 3
    LATERAL_MOVEMENT = 4
    EXFILTRATION = 5

class RedTeamEngine:
    def __init__(self):
        self.current_phase = AttackPhase.RECONNAISSANCE
        self.sophistication = 0.8  # APT-level
        self.detected = False
        self.blue_team_response_level = 0.0  # 0.0-1.0

        self.dbe_socket = DBESocket("/var/run/dsmil/redteam-engine.sock")

        logger.info(f"Red Team Engine initialized (Device {DEVICE_ID})")

    def execute_attack(self, action: str, payload: Dict):
        """Execute red team action with adaptive tactics"""

        if action == "phishing_email":
            success_prob = payload.get("success_probability", 0.3)

            # Adapt based on blue team response
            if self.blue_team_response_level > 0.7:
                # Blue team is alert, use more sophisticated phishing
                success_prob *= 0.5
                logger.info("Blue team alert detected, reducing phishing success probability")

            # Simulate user click
            if random.random() < success_prob:
                logger.warning(f"PHISHING SUCCESS: User {payload['target_user']} clicked malicious link")
                self.current_phase = AttackPhase.INITIAL_ACCESS
                self._inject_malware_beacon()
            else:
                logger.info(f"Phishing failed: User {payload['target_user']} did not click")

        elif action == "lateral_movement":
            if self.detected:
                # Switch to stealthier technique
                logger.info("Detection active, switching to WMI-based lateral movement")
                technique = "wmi_exec"
            else:
                technique = "psexec"

            self._inject_lateral_movement(technique)

        elif action == "data_exfiltration":
            if self.blue_team_response_level > 0.5:
                # Use DNS tunneling to evade detection
                logger.info("High blue team response, using DNS tunneling for exfiltration")
                self._inject_dns_tunnel()
            else:
                # Direct HTTPS exfiltration
                self._inject_https_exfiltration()

    def _inject_malware_beacon(self):
        """Inject C2 beacon traffic (SIGINT event)"""
        beacon_data = {
            "source_ip": "10.0.1.45",  # Compromised host
            "dest_ip": "203.0.113.45",  # C2 server
            "protocol": "HTTPS",
            "port": 443,
            "beacon_interval_seconds": 300,  # 5 minutes
            "timestamp": time.time()
        }

        msg = DBEMessage(
            msg_type=0x93,  # INJECT_EVENT
            device_id_src=DEVICE_ID,
            device_id_dst=65,  # SIGINT Injector
            tlvs={
                "INJECT_TYPE": "SIGINT",
                "EVENT_TYPE": "c2_beacon",
                "PAYLOAD": str(beacon_data),
                "RED_TEAM_ACTION": "initial_access"
            }
        )

        self.dbe_socket.send_to("/var/run/dsmil/event-injector-65.sock", msg)
        logger.warning("Injected C2 beacon traffic")

if __name__ == "__main__":
    engine = RedTeamEngine()
    # Wait for RED_TEAM_ACTION messages
```

---

## 7. Device 70: After-Action Report Generator

**Purpose:** Automated metrics collection and visualization for post-exercise analysis.

**Token IDs:**
- `0x80D2` (STATUS): Report generation progress
- `0x80D3` (CONFIG): Report template, output format
- `0x80D4` (DATA): Collected metrics, decision trees

**AAR Components:**

1. **Executive Summary:**
   - Exercise duration, participants, objectives achieved
   - Key findings and recommendations
   - Classification and distribution list

2. **Timeline Reconstruction:**
   - All injected events with timestamps
   - Blue team responses and actions taken
   - Red team attack progression
   - Decision points and outcomes

3. **Performance Metrics:**
   - **Response Times:** Time from event injection to detection, analysis, containment
   - **Decision Accuracy:** L6/L7 predictions vs actual outcomes
   - **Threat Identification:** True positives, false positives, false negatives
   - **Operator Performance:** Individual analyst scores, SOC team coordination

4. **Decision Tree Visualization:**
   - L7-L9 reasoning chains displayed as flowcharts
   - Show which intelligence informed each decision
   - Highlight decision bottlenecks and delays

5. **SHRINK Stress Analysis:**
   - Operator cognitive load over time
   - Decision fatigue indicators
   - High-stress periods correlated with event density
   - Recommendations for shift scheduling and breaks

6. **Lessons Learned:**
   - What worked well
   - What needs improvement
   - Gaps in capability or training
   - Recommendations for future exercises

**Output Formats:**
- **PDF:** Executive summary, charts, timeline (for briefings)
- **HTML:** Interactive dashboard with drill-down capability
- **JSON:** Machine-readable data for trend analysis across exercises

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/aar_generator.py
"""
DSMIL After-Action Report Generator (Device 70)
Automated metrics and visualization for post-exercise analysis
"""

import time
import json
import logging
import psycopg2
import redis
from typing import Dict, List
from dataclasses import dataclass

from dsmil_dbe import DBEMessage, DBESocket

DEVICE_ID = 70
TOKEN_BASE = 0x80D2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExerciseMetrics:
    total_events: int
    detection_rate: float
    mean_response_time_seconds: float
    false_positive_rate: float
    objectives_achieved: int
    objectives_total: int

class AARGenerator:
    def __init__(self):
        self.redis = redis.Redis(host="localhost", db=15)  # Exercise DB
        self.pg = psycopg2.connect(
            host="localhost",
            database="exercise_db",
            user="dsmil_exercise",
            password="<from-vault>"
        )

        self.dbe_socket = DBESocket("/var/run/dsmil/aar-generator.sock")

        logger.info(f"AAR Generator initialized (Device {DEVICE_ID})")

    def generate_aar(self, request: DBEMessage) -> str:
        """Generate comprehensive after-action report"""
        tenant_id = request.tlv_get("EXERCISE_TENANT_ID")
        scenario_id = request.tlv_get("SCENARIO_ID")
        start_time = float(request.tlv_get("START_TIME"))
        end_time = float(request.tlv_get("END_TIME"))

        logger.info(f"Generating AAR for {tenant_id}, Scenario: {scenario_id}")

        # Collect metrics from Postgres
        metrics = self._collect_metrics(tenant_id, start_time, end_time)

        # Reconstruct timeline
        timeline = self._reconstruct_timeline(tenant_id)

        # Analyze decision trees (from L7-L9 logs)
        decision_trees = self._analyze_decision_trees(tenant_id)

        # SHRINK stress analysis (from operator metrics)
        shrink_analysis = self._shrink_analysis(tenant_id)

        # Build report
        report = {
            "tenant_id": tenant_id,
            "scenario_id": scenario_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_hours": (end_time - start_time) / 3600,
            "metrics": metrics.__dict__,
            "timeline": timeline,
            "decision_trees": decision_trees,
            "shrink_analysis": shrink_analysis,
            "generated_at": time.time()
        }

        # Save to file
        output_path = f"/var/log/dsmil/aar_{tenant_id}_{scenario_id}.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"AAR generated: {output_path}")

        # TODO: Generate PDF and HTML versions

        return output_path

    def _collect_metrics(self, tenant_id: str, start_time: float, end_time: float) -> ExerciseMetrics:
        """Collect performance metrics from database"""
        with self.pg.cursor() as cur:
            # Total events injected
            cur.execute(f"""
                SELECT COUNT(*) FROM {tenant_id}_events
                WHERE timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
                  AND event_type = 'INJECT_EVENT'
            """, (start_time, end_time))
            total_events = cur.fetchone()[0]

            # Detection rate (events that triggered L3 alerts)
            cur.execute(f"""
                SELECT COUNT(*) FROM {tenant_id}_events
                WHERE timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
                  AND event_type = 'L3_ALERT'
            """, (start_time, end_time))
            detected_events = cur.fetchone()[0]
            detection_rate = detected_events / total_events if total_events > 0 else 0.0

            # Mean response time (inject to detection)
            cur.execute(f"""
                SELECT AVG(EXTRACT(EPOCH FROM (alert.timestamp - inject.timestamp)))
                FROM {tenant_id}_events inject
                JOIN {tenant_id}_events alert
                  ON inject.payload->>'event_id' = alert.payload->>'correlated_event_id'
                WHERE inject.event_type = 'INJECT_EVENT'
                  AND alert.event_type = 'L3_ALERT'
                  AND inject.timestamp BETWEEN to_timestamp(%s) AND to_timestamp(%s)
            """, (start_time, end_time))
            mean_response_time = cur.fetchone()[0] or 0.0

        return ExerciseMetrics(
            total_events=total_events,
            detection_rate=detection_rate,
            mean_response_time_seconds=mean_response_time,
            false_positive_rate=0.0,  # TODO: Calculate
            objectives_achieved=0,  # TODO: Parse from scenario
            objectives_total=0
        )

    def _reconstruct_timeline(self, tenant_id: str) -> List[Dict]:
        """Reconstruct exercise timeline from events"""
        with self.pg.cursor() as cur:
            cur.execute(f"""
                SELECT timestamp, event_type, device_id, payload
                FROM {tenant_id}_events
                ORDER BY timestamp ASC
            """)

            timeline = []
            for row in cur.fetchall():
                timeline.append({
                    "timestamp": row[0].isoformat(),
                    "event_type": row[1],
                    "device_id": row[2],
                    "payload": row[3]
                })

        return timeline

    def _analyze_decision_trees(self, tenant_id: str) -> List[Dict]:
        """Analyze L7-L9 decision reasoning chains"""
        # TODO: Query L7/L8/L9 logs for decision chains
        return []

    def _shrink_analysis(self, tenant_id: str) -> Dict:
        """SHRINK stress analysis for operator cognitive load"""
        # TODO: Analyze operator metrics (response times, errors, fatigue indicators)
        return {
            "peak_stress_time": None,
            "mean_cognitive_load": 0.5,
            "fatigue_indicators": []
        }

if __name__ == "__main__":
    generator = AARGenerator()
    # Wait for AAR_REQUEST messages
```

---

## 8. Security & Authorization

### 8.1 Exercise Data Segregation

**Redis Schema Isolation:**
- Exercise data in DB 15 (separate from operational DB 0)
- Keys prefixed with `exercise:{tenant_id}:*`
- Flush DB 15 after exercise completion and AAR

**Postgres Schema Isolation:**
- Separate database: `exercise_db`
- Tenant-specific tables: `{tenant_id}_events`, `{tenant_id}_metrics`
- Drop tables after retention period (90 days SECRET, 1 year ATOMAL)

### 8.2 ATOMAL Exercise Authorization

**Two-Person Integrity:**
- ATOMAL exercises require dual ML-DSA-87 signatures from different authorized personnel
- Signatures verified against whitelist of authorized exercise directors
- Both signatures logged in audit trail

**Access Control:**
- ATOMAL exercise data accessible only to NATO SECRET clearance holders
- Need-to-know enforcement via DBE `COMPARTMENT_MASK`
- Export restrictions: REL NATO markings enforced

### 8.3 ROE Enforcement

**TRAINING Mode Safety:**
- Global `ROE_LEVEL=TRAINING` set for all L3-L9 devices during exercise
- Device 61 (NC3 Integration) **disabled** to prevent kinetic outputs
- L9 Executive layer limited to analysis-only (no command issuance)

**Post-Exercise Restoration:**
- Operational ROE levels restored after exercise stop
- Device 61 re-enabled with audit logging
- Verification checks before returning to operational status

---

## 9. Implementation Details

### 9.1 Docker Compose Configuration

```yaml
# /opt/dsmil/docker-compose-phase10.yml
version: '3.8'

services:
  exercise-controller:
    image: dsmil/exercise-controller:1.0
    container_name: dsmil-exercise-controller-63
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - /var/log/dsmil:/var/log/dsmil
    environment:
      - DEVICE_ID=63
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  scenario-engine:
    image: dsmil/scenario-engine:1.0
    container_name: dsmil-scenario-engine-64
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - /opt/dsmil/scenarios:/scenarios:ro
    environment:
      - DEVICE_ID=64
    restart: unless-stopped

  sigint-injector:
    image: dsmil/event-injector:1.0
    container_name: dsmil-sigint-injector-65
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=65
      - INJECT_TYPE=SIGINT
    restart: unless-stopped

  imint-injector:
    image: dsmil/event-injector:1.0
    container_name: dsmil-imint-injector-66
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=66
      - INJECT_TYPE=IMINT
    restart: unless-stopped

  humint-injector:
    image: dsmil/event-injector:1.0
    container_name: dsmil-humint-injector-67
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=67
      - INJECT_TYPE=HUMINT
    restart: unless-stopped

  redteam-engine:
    image: dsmil/redteam-engine:1.0
    container_name: dsmil-redteam-engine-68
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=68
    restart: unless-stopped

  blueforce-sim:
    image: dsmil/blueforce-sim:1.0
    container_name: dsmil-blueforce-sim-69
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=69
    restart: unless-stopped

  aar-generator:
    image: dsmil/aar-generator:1.0
    container_name: dsmil-aar-generator-70
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - /var/log/dsmil:/var/log/dsmil
    environment:
      - DEVICE_ID=70
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  training-assess:
    image: dsmil/training-assess:1.0
    container_name: dsmil-training-assess-71
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=71
    restart: unless-stopped

  exercise-recorder:
    image: dsmil/exercise-recorder:1.0
    container_name: dsmil-exercise-recorder-72
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - /var/log/dsmil/recordings:/recordings
    environment:
      - DEVICE_ID=72
      - STORAGE_PATH=/recordings
    restart: unless-stopped

networks:
  default:
    name: dsmil-exercise-net
```

### 9.2 Health Check Endpoints

All Phase 10 services expose health checks via DBE protocol:

```python
# Health check request
msg = DBEMessage(
    msg_type=0x96,  # EXERCISE_STATUS
    device_id_src=0,
    device_id_dst=63,  # Exercise Controller
    tlvs={"COMMAND": "health_check"}
)

# Health check response
response = {
    "status": "OK",  # OK, DEGRADED, FAILED
    "device_id": 63,
    "uptime_seconds": 3600,
    "memory_usage_mb": 180,
    "last_activity": time.time()
}
```

---

## 10. Testing & Validation

### 10.1 Unit Tests

```python
#!/usr/bin/env python3
# tests/test_exercise_controller.py
"""
Unit tests for Exercise Controller (Device 63)
"""

import unittest
from exercise_controller import ExerciseController, ExerciseTenant

class TestExerciseController(unittest.TestCase):

    def setUp(self):
        self.controller = ExerciseController()

    def test_dual_auth_validation(self):
        """Test two-person authorization for ATOMAL exercises"""
        # Valid case: two different signatures
        tenant = ExerciseTenant(
            tenant_id="ATOMAL_EXERCISE",
            classification="ATOMAL",
            scenario_id="test-001",
            start_time=time.time(),
            participants=[],
            dual_auth_required=True,
            auth_signature_1=b"sig1_from_director_A",
            auth_signature_2=b"sig2_from_director_B"
        )

        result = self.controller._validate_dual_auth(tenant)
        self.assertTrue(result)

    def test_roe_enforcement(self):
        """Test ROE_LEVEL=TRAINING enforcement"""
        self.controller._set_global_roe("TRAINING")

        # Verify all L3-L9 devices have TRAINING ROE
        for device_id in range(14, 63):
            roe = self.controller.redis.get(f"device:{device_id}:roe_level")
            self.assertEqual(roe, "TRAINING")

    def test_nc3_disable_during_exercise(self):
        """Test Device 61 (NC3) disabled during exercise"""
        self.controller._disable_nc3()

        enabled = self.controller.redis.get("device:61:enabled")
        self.assertEqual(enabled, "false")

if __name__ == '__main__':
    unittest.main()
```

### 10.2 Integration Tests

```bash
#!/bin/bash
# tests/integration/test_full_exercise.sh
# Integration test: Run full exercise from start to AAR

set -e

echo "[TEST] Starting full exercise integration test..."

# 1. Start all Phase 10 services
docker-compose -f /opt/dsmil/docker-compose-phase10.yml up -d

# 2. Load test scenario
SCENARIO_PATH="/opt/dsmil/scenarios/test-cyber-attack.json"

# 3. Start exercise (with dual auth for ATOMAL)
# Generate two signatures (mock)
SIG1=$(echo "test-sig-1" | base64)
SIG2=$(echo "test-sig-2" | base64)

curl -X POST http://localhost:8080/exercise/start \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "ATOMAL_EXERCISE",
    "scenario_path": "'$SCENARIO_PATH'",
    "classification": "ATOMAL",
    "dual_auth_sig_1": "'$SIG1'",
    "dual_auth_sig_2": "'$SIG2'"
  }'

# 4. Wait for scenario to execute (10 minutes)
echo "[TEST] Waiting for scenario execution (10 min)..."
sleep 600

# 5. Stop exercise
curl -X POST http://localhost:8080/exercise/stop

# 6. Wait for AAR generation
echo "[TEST] Waiting for AAR generation..."
sleep 60

# 7. Verify AAR file exists
AAR_FILE="/var/log/dsmil/aar_ATOMAL_EXERCISE_*.json"
if [ ! -f $AAR_FILE ]; then
    echo "[TEST] FAILED: AAR file not found"
    exit 1
fi

echo "[TEST] AAR generated: $AAR_FILE"

# 8. Verify metrics in AAR
TOTAL_EVENTS=$(jq '.metrics.total_events' $AAR_FILE)
if [ "$TOTAL_EVENTS" -eq 0 ]; then
    echo "[TEST] FAILED: No events recorded"
    exit 1
fi

echo "[TEST] SUCCESS: $TOTAL_EVENTS events recorded and analyzed"

# 9. Cleanup
docker-compose -f /opt/dsmil/docker-compose-phase10.yml down

echo "[TEST] Full exercise integration test PASSED"
```

### 10.3 Red Team Exercise Scenarios

**Scenario 1: APT Cyber Attack**
- Duration: 4 hours
- Events: 50+ synthetic SIGINT/IMINT events
- Red Team: APT-style adversary with persistence
- Objectives: Detect recon, identify C2, contain lateral movement

**Scenario 2: Insider Threat**
- Duration: 2 hours
- Events: 20+ HUMINT/SIGINT events
- Red Team: Malicious insider with valid credentials
- Objectives: Detect anomalous access, prevent data exfiltration

**Scenario 3: Multi-Domain Coalition Exercise**
- Duration: 8 hours
- Events: 100+ SIGINT/IMINT/HUMINT events
- Red Team: Nation-state adversary with cyber + physical capabilities
- Objectives: NATO interoperability, ATOMAL information sharing

---

## 11. Exit Criteria

Phase 10 is considered complete when:

- [ ] All 10 devices (63-72) operational and health-check passing
- [ ] Successful 24-hour exercise with 10,000+ synthetic events injected
- [ ] ATOMAL exercise completed with dual authorization verified
- [ ] After-action report generated within 1 hour of exercise completion
- [ ] Red team scenario with adaptive tactics demonstrated (3 tactic changes observed)
- [ ] Exercise data segregation verified (no operational data contamination)
- [ ] ROE enforcement tested (Device 61 NC3 disabled, no kinetic outputs)
- [ ] Full message replay from Exercise Recorder (Device 72) functional
- [ ] Integration tests passing with 95%+ success rate
- [ ] Documentation complete (operator manuals, scenario templates)

---

## 12. Future Enhancements

**Post-Phase 10 Capabilities:**

1. **AI-Powered Red Team:** L7 LLM-driven adversary with creative tactics
2. **VR/AR Exercise Visualization:** Immersive 3D battlefield representation
3. **Multi-Site Distributed Exercises:** Federated DSMIL instances across locations
4. **Exercise-as-Code:** Git-versioned scenario definitions with CI/CD
5. **Automated Scenario Generation:** L7-generated scenarios based on threat intelligence

---

**End of Phase 10 Specification**
