# DSMIL Complete System Reference
**Last Updated**: 2025-11-07
**System**: Dell Latitude 5450 MIL-SPEC JRTC1
**Status**: Production Ready - 84 Devices Active

---

## Quick Facts

| Specification | Value |
|--------------|-------|
| **Total Devices** | 84 |
| **Device Groups** | 7 (Groups 0-6) |
| **Devices per Group** | 12 |
| **Safe Devices** | 6 |
| **Quarantined Devices** | 5 |
| **Risky/Unknown** | 73 |
| **Control Subsystems** | 9 |
| **Memory Token Range** | 0x8000-0x806B |

---

## Device Groups (7 Groups × 12 Devices = 84 Total)

### Group 0: Core Security & Emergency (0x8000-0x800B)
**Purpose**: Foundation security infrastructure
**Risk Level**: CRITICAL
**Dependencies**: None (root group)

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8000 | DSMIL0D0 | Master Controller | Safe |
| 0x8001 | DSMIL0D1 | Cryptographic Engine | Monitored |
| 0x8002 | DSMIL0D2 | Secure Key Storage | Monitored |
| 0x8003 | DSMIL0D3 | Authentication Module | Monitored |
| 0x8004 | DSMIL0D4 | Access Control | Monitored |
| 0x8005 | DSMIL0D5 | Audit Logger | Safe |
| 0x8006 | DSMIL0D6 | Integrity Monitor | Monitored |
| 0x8007 | DSMIL0D7 | Secure Boot Controller | Monitored |
| 0x8008 | DSMIL0D8 | TPM Interface | Safe |
| 0x8009 | DSMIL0D9 | Emergency Wipe | **QUARANTINED** |
| 0x800A | DSMIL0DA | Recovery Controller | **QUARANTINED** |
| 0x800B | DSMIL0DB | Hidden Memory Controller | **QUARANTINED** |

**⚠️ WARNING**: 3 devices in Group 0 are QUARANTINED (data destruction capability)

### Group 1: Extended Security (0x8010-0x801B)
**Purpose**: Advanced threat detection and prevention
**Risk Level**: HIGH
**Dependencies**: Requires Group 0

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8010 | DSMIL1D0 | Group 1 Controller | Safe |
| 0x8011 | DSMIL1D1 | Threat Detection | Monitored |
| 0x8012 | DSMIL1D2 | Intrusion Prevention | Monitored |
| 0x8013 | DSMIL1D3 | Network Security | Monitored |
| 0x8014 | DSMIL1D4 | Malware Scanner | Monitored |
| 0x8015 | DSMIL1D5 | Behavioral Analysis | Monitored |
| 0x8016 | DSMIL1D6 | Security Policy Engine | Monitored |
| 0x8017 | DSMIL1D7 | Incident Response | Monitored |
| 0x8018 | DSMIL1D8 | Forensics Module | Monitored |
| 0x8019 | DSMIL1D9 | Network Kill Switch | **QUARANTINED** |
| 0x801A | DSMIL1DA | Vulnerability Scanner | Monitored |
| 0x801B | DSMIL1DB | Security Analytics | Monitored |

**⚠️ WARNING**: 1 device in Group 1 is QUARANTINED (network kill capability)

### Group 2: Network & Communications (0x8020-0x802B)
**Purpose**: Network management and tactical communications
**Risk Level**: MEDIUM
**Dependencies**: Requires Groups 0, 1

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8020 | DSMIL2D0 | Network Controller | Safe |
| 0x8021 | DSMIL2D1 | Ethernet Manager | Monitored |
| 0x8022 | DSMIL2D2 | WiFi Controller | Monitored |
| 0x8023 | DSMIL2D3 | Bluetooth Manager | Monitored |
| 0x8024 | DSMIL2D4 | VPN Engine | Monitored |
| 0x8025 | DSMIL2D5 | Firewall | Monitored |
| 0x8026 | DSMIL2D6 | QoS Manager | Monitored |
| 0x8027 | DSMIL2D7 | Network Monitor | Monitored |
| 0x8028 | DSMIL2D8 | DNS Security | Monitored |
| 0x8029 | DSMIL2D9 | Communications Blackout | **QUARANTINED** |
| 0x802A | DSMIL2DA | Router Functions | Monitored |
| 0x802B | DSMIL2DB | Network Storage | Monitored |

**⚠️ WARNING**: 1 device in Group 2 is QUARANTINED (communications blackout)

### Group 3: Data Processing (0x8030-0x803B)
**Purpose**: Information processing and management
**Risk Level**: MEDIUM
**Dependencies**: Requires Group 0

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8030 | DSMIL3D0 | Processing Controller | Safe |
| 0x8031-0x803B | DSMIL3D1-3DB | Data Processing Functions | Monitored |

### Group 4: Storage Control (0x8040-0x804B)
**Purpose**: Storage management and control
**Risk Level**: MEDIUM
**Dependencies**: Requires Groups 0, 3

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8040-0x804B | DSMIL4D0-4DB | Storage Functions | Monitored |

### Group 5: Peripheral Management (0x8050-0x805B)
**Purpose**: Peripheral device management
**Risk Level**: MEDIUM-LOW
**Dependencies**: Requires Group 0

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8050-0x805B | DSMIL5D0-5DB | Peripheral Functions | Monitored |

### Group 6: Training Functions (0x8060-0x806B)
**Purpose**: JRTC1 training and simulation
**Risk Level**: LOW
**Dependencies**: Optional (training mode)

| Device ID | Name | Function | Status |
|-----------|------|----------|--------|
| 0x8060-0x806B | DSMIL6D0-6DB | Training/Simulation Functions | Monitored |

---

## Control Subsystems (9 Total)

The DSMIL platform is controlled through 9 specialized subsystems implemented in `dsmil_subsystem_controller.py`:

### 1. DEVICE_CONTROL
**Purpose**: Core device activation and control
**Functions**: Safe device activation, status monitoring, device health
**Safe Devices Available**: 6
- DSMIL0D0 (Master Controller)
- DSMIL0D5 (Audit Logger)
- DSMIL0D8 (TPM Interface)
- DSMIL1D0 (Group 1 Controller)
- DSMIL2D0 (Network Controller)
- DSMIL3D0 (Processing Controller)

### 2. MONITORING
**Purpose**: Real-time system and device monitoring
**Functions**: Device state tracking, health checks, performance metrics
**Status**: Operational

### 3. SECURITY
**Purpose**: Security enforcement and quarantine management
**Functions**: Quarantine enforcement, access control, threat detection
**Status**: Operational
**Quarantined Devices**: 5 (absolutely prohibited)
- 0x8009: Emergency Wipe (data destruction)
- 0x800A: Recovery Controller (data destruction)
- 0x800B: Hidden Memory Controller (data destruction)
- 0x8019: Network Kill Switch (network termination)
- 0x8029: Communications Blackout (communications termination)

### 4. THERMAL
**Purpose**: Thermal monitoring and thermal impact tracking
**Functions**: CPU temperature monitoring, thermal throttling, device thermal impact
**Status**: Operational
**Threshold**: 80°C warning, 90°C critical

### 5. TPM_ATTESTATION
**Purpose**: TPM 2.0 hardware attestation and verification
**Functions**: Platform integrity quotes, PCR measurements, hardware crypto
**Status**: Operational
**Algorithms Supported**: 88 cryptographic algorithms
**Hardware**: STMicroelectronics/Infineon TPM 2.0

### 6. AVX512_UNLOCK
**Purpose**: AVX-512 instruction set unlock status
**Functions**: Check AVX-512 availability, unlock status monitoring
**Status**: Operational
**CPU**: Intel Core Ultra 7 (Meteor Lake) with AI Boost

### 7. NPU_STATUS
**Purpose**: Neural Processing Unit status and capabilities
**Functions**: NPU detection, performance monitoring, AI acceleration status
**Status**: Operational
**Hardware**: 48 TOPS NPU (AI Boost)
**Military Mode**: Detectable

### 8. GNA_PRESENCE
**Purpose**: Gaussian Neural Accelerator detection
**Functions**: GNA hardware detection, acceleration status
**Status**: Operational
**Hardware**: Intel GNA integrated

### 9. MODE5_INTEGRITY
**Purpose**: Mode 5 platform integrity enforcement
**Functions**: Integrity verification, trusted boot, platform security
**Status**: Operational
**Mode**: JRTC1 Training Variant

---

## Safety Architecture

### 4-Layer Protection

1. **Module Constants** (`dsmil_device_database.py`)
   - QUARANTINED_DEVICES = hardcoded list
   - Immutable at runtime

2. **Controller Methods** (`dsmil_subsystem_controller.py`)
   - Check device against QUARANTINED_DEVICES before ANY operation
   - Reject quarantined devices immediately

3. **Activation Checks**
   - Verify device in SAFE_DEVICES before activation
   - Log all activation attempts

4. **API Responses**
   - Return error for quarantined device access
   - Audit log all rejected attempts

### Emergency Procedures

**If quarantined device activation attempted:**
1. Immediate rejection at controller level
2. Log security event
3. Alert dashboard
4. NO activation under ANY circumstances

**System integrity verification:**
```bash
# Check quarantine enforcement
curl http://localhost:5050/api/dsmil/subsystems | jq '.quarantined_devices'

# Verify safe devices only
curl http://localhost:5050/api/dsmil/safe-devices
```

---

## Access Methods

### SMI Interface (Primary)
```c
// I/O Port Access (requires root/iopl(3))
outw(token_id, 0x164E);     // Write device token
status = inb(0x164F);       // Read device status
```

### Memory Mapping (Read-Only)
```
Base Address: 0x60000000
Size: 360MB (chunked access recommended)
Access: Read-only for safety
```

### Python API
```python
from dsmil_subsystem_controller import DSMILSubsystemController

controller = DSMILSubsystemController()

# Get all subsystems
subsystems = controller.get_all_subsystems_status()

# Safe device activation only
result = controller.activate_safe_device(0x8000)  # Master Controller

# Quarantined device (REJECTED)
result = controller.activate_safe_device(0x8009)  # ERROR: Quarantined
```

---

## Integration Points

### 1. Dashboard (ai_gui_dashboard.py)
- Real-time device status display
- Subsystem health monitoring
- Safe device activation interface
- Quarantine enforcement visualization

### 2. Quantum Crypto Layer (quantum_crypto_layer.py)
- TPM 2.0 hardware integration
- 88 algorithm support
- CSNA 2.0 compliance
- Perfect forward secrecy

### 3. API Endpoints
- `/api/dsmil/health` - System health
- `/api/dsmil/subsystems` - All subsystem status
- `/api/dsmil/devices` - All 84 devices
- `/api/dsmil/safe-devices` - 6 safe devices only
- `/api/dsmil/activate` - Safe device activation

---

## Current Status Summary

✅ **Complete**:
- 84 devices discovered and mapped
- 9 subsystems operational
- 6 safe devices available for activation
- 5 quarantined devices absolutely blocked
- TPM 2.0 integration active
- Quantum encryption operational
- Dashboard fully functional

⚠️ **Limitations**:
- 73 devices remain risky/unknown (not activated)
- Full device function mapping incomplete
- Some group dependencies untested

🔒 **Security**:
- 4-layer quarantine enforcement
- Hardware attestation active
- CSNA 2.0 compliant
- Zero destructive device activation possible

---

## Documentation Hierarchy

**Primary References** (Use These):
1. This document (DSMIL_CURRENT_REFERENCE.md)
2. `02-ai-engine/dsmil_device_database.py` - Device database
3. `02-ai-engine/dsmil_subsystem_controller.py` - Subsystem implementation
4. `README.md` - Quick start guide

**Historical/Deprecated** (Reference Only):
- DEPRECATED-DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md
- DEPRECATED-DSMIL-DEVICE-FUNCTION-ANALYSIS.md

**Technical Details**:
- EXECUTIVE_SUMMARY.md - Discovery story
- DSMIL_INTEGRATION_SUCCESS.md - Kernel integration
- DSMIL_COMPATIBILITY_REPORT.md - Compatibility analysis

---

## Quick Command Reference

```bash
# Start Dashboard
./scripts/start-dashboard.sh
# Access: http://localhost:5050

# Check Device Status
curl http://localhost:5050/api/dsmil/devices | jq '.'

# View Subsystems
curl http://localhost:5050/api/dsmil/subsystems | jq '.'

# Get TPM Status
curl http://localhost:5050/api/tpm/status | jq '.'

# Check NPU
curl http://localhost:5050/api/dsmil/subsystems | jq '.npu_status'

# Run All Tests
curl -X POST http://localhost:5050/api/benchmark/run
```

---

**For Support**: See [DEPLOYMENT_READY.md](../../DEPLOYMENT_READY.md) for complete setup and troubleshooting.

**Last Verified**: 2025-11-07
**System Version**: DSMIL 8.3.1
**Platform**: Dell Latitude 5450 JRTC1 MIL-SPEC
