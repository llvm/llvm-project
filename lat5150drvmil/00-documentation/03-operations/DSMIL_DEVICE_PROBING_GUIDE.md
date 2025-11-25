# DSMIL Device Function Probing Guide
**Last Updated**: 2025-11-07
**Purpose**: Comprehensive guide to safely probe and discover DSMIL device functionality
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## Executive Summary

This guide provides step-by-step procedures for safely discovering the functions and capabilities of all 84 DSMIL devices. It includes:
- Current TODO list from the codebase
- Easy wins for quick improvements
- 5-phase progressive probing methodology
- Safety protocols and quarantine enforcement
- Testing procedures for each device group

**Status**: 6 devices verified safe, 5 quarantined, 73 awaiting function discovery

---

## Current TODO List from Codebase

### High Priority TODOs

#### 1. Device Activation Implementation
**File**: `02-ai-engine/dsmil_subsystem_controller.py:550`
```python
# TODO: Implement actual device activation via sysfs/ioctl
# For now, return success for safe devices
```

**Impact**: HIGH - Core functionality
**Effort**: MEDIUM (2-4 hours)
**Blocker**: Need sysfs interface or ioctl mechanism

**Implementation Steps**:
1. Check if kernel module exposes sysfs interface (`/sys/class/dsmil/`)
2. If not, implement ioctl calls via `/dev/dsmil0`
3. Test with safe device (0x8000) first
4. Add error handling and rollback
5. Integrate with subsystem controller

#### 2. Thermal Monitoring
**File**: `01-source/kernel/dsmil_hal.c:727`
```c
/* TODO: Implement actual thermal reading */
```

**Impact**: MEDIUM - Safety feature
**Effort**: LOW (1-2 hours)
**Blocker**: None

**Implementation Steps**:
1. Read from `/sys/class/thermal/thermal_zone*/temp`
2. Map thermal zones to CPU cores
3. Add threshold alerts (80°C warning, 90°C critical)
4. Log thermal events to audit framework

#### 3. TPM State Tracking
**Files**:
- `01-source/kernel-driver/dell-millspec-enhanced.c:1725`
- `01-source/kernel-driver/dell-millspec-enhanced.c:1732`

```c
status.tpm_measured = 0; /* TODO: track TPM state */
status.event_count = 0; /* TODO: track event count */
```

**Impact**: MEDIUM - Attestation quality
**Effort**: LOW (1-2 hours)
**Blocker**: None

**Implementation Steps**:
1. Use `tpm2_pcrread` to track PCR measurements
2. Count events in TPM event log (`/sys/kernel/security/tpm0/binary_bios_measurements`)
3. Store state in subsystem controller
4. Expose via API endpoint

### Medium Priority TODOs

#### 4. Event Logging with Trace Infrastructure
**Files**:
- `01-source/kernel-driver/dell-millspec-enhanced.c:235`
- `01-source/kernel-driver/dell-millspec-enhanced.c:2611`

**Impact**: LOW-MEDIUM - Debugging capability
**Effort**: MEDIUM (3-5 hours)
**Blocker**: Kernel trace infrastructure

#### 5. Authorization Checking
**File**: `01-source/kernel/dsmil_hal.c:799`
```c
/* TODO: Implement proper authorization checking */
```

**Impact**: MEDIUM - Security enhancement
**Effort**: MEDIUM (2-3 hours)
**Blocker**: MFA auth framework integration

#### 6. Audit Framework Storage
**File**: `01-source/kernel/dsmil_audit_framework.c:1064`
```c
/* TODO: Implement actual storage write */
```

**Impact**: LOW - Compliance feature
**Effort**: LOW (1-2 hours)
**Blocker**: None

---

## Easy Wins (Quick Improvements)

### 🎯 Win #1: Implement Thermal Monitoring (1-2 hours)
**Value**: Immediate safety improvement
**Risk**: None (read-only operation)

**Steps**:
```python
# Add to dsmil_subsystem_controller.py

def get_thermal_status_enhanced(self):
    """Enhanced thermal monitoring with per-core readings"""
    temps = []
    for zone in Path('/sys/class/thermal/').glob('thermal_zone*'):
        temp_file = zone / 'temp'
        type_file = zone / 'type'
        if temp_file.exists() and type_file.exists():
            temp = int(temp_file.read_text().strip()) / 1000.0
            zone_type = type_file.read_text().strip()
            temps.append({
                'zone': zone.name,
                'type': zone_type,
                'temp_c': temp,
                'status': 'critical' if temp > 90 else 'warning' if temp > 80 else 'normal'
            })
    return temps
```

### 🎯 Win #2: Add TPM PCR Tracking (1-2 hours)
**Value**: Better attestation visibility
**Risk**: None (read-only operation)

**Steps**:
```python
# Add to dsmil_subsystem_controller.py

def get_tpm_pcr_state(self):
    """Get current TPM PCR measurements"""
    try:
        result = subprocess.run(
            ['tpm2_pcrread', 'sha256:0,1,2,3,4,5,6,7,8'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse PCR values
            pcrs = {}
            for line in result.stdout.split('\n'):
                if 'sha256:' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        pcr_num = parts[0].strip()
                        pcr_val = parts[1].strip()
                        pcrs[pcr_num] = pcr_val
            return pcrs
    except Exception as e:
        return {'error': str(e)}
    return {}
```

### 🎯 Win #3: Device Status Caching (30 minutes)
**Value**: Reduce repeated SMI calls, improve performance
**Risk**: None

**Steps**:
```python
# Add to dsmil_subsystem_controller.py

from datetime import datetime, timedelta

class DSMILSubsystemController:
    def __init__(self):
        # ... existing init ...
        self.device_status_cache = {}
        self.cache_ttl = timedelta(seconds=5)

    def get_device_status_cached(self, device_id: int):
        """Get device status with caching"""
        cache_key = f"status_{device_id:04X}"
        now = datetime.now()

        if cache_key in self.device_status_cache:
            cached_time, cached_value = self.device_status_cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_value

        # Fetch fresh status
        status = self.get_device_status(device_id)
        self.device_status_cache[cache_key] = (now, status)
        return status
```

### 🎯 Win #4: Add Device Operation History (1 hour)
**Value**: Track what operations have been attempted, debugging
**Risk**: None

**Steps**:
```python
# Add to dsmil_subsystem_controller.py

from collections import deque

class DSMILSubsystemController:
    def __init__(self):
        # ... existing init ...
        self.operation_history = deque(maxlen=1000)

    def log_operation(self, device_id: int, operation: str, success: bool, details: str = ""):
        """Log device operation for history tracking"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': f"0x{device_id:04X}",
            'operation': operation,
            'success': success,
            'details': details
        }
        self.operation_history.append(entry)

    def get_operation_history(self, device_id: Optional[int] = None, limit: int = 100):
        """Get operation history, optionally filtered by device"""
        history = list(self.operation_history)
        if device_id:
            history = [h for h in history if h['device_id'] == f"0x{device_id:04X}"]
        return history[-limit:]
```

### 🎯 Win #5: Add Subsystem Health Scores (30 minutes)
**Value**: Quick overview of system health
**Risk**: None

**Steps**:
```python
# Add to dsmil_subsystem_controller.py

def get_subsystem_health_score(self) -> Dict[str, float]:
    """Calculate health score for each subsystem (0.0 - 1.0)"""
    scores = {}
    for subsystem_type in SubsystemType:
        status = self.get_subsystem_status(subsystem_type)
        if not status:
            scores[subsystem_type.value] = 0.0
            continue

        # Base score on operational status
        score = 1.0 if status.operational else 0.0

        # Adjust for specific subsystem factors
        if subsystem_type == SubsystemType.THERMAL:
            temp = status.status_info.get('current_temp', 0)
            if temp > 90:
                score *= 0.5
            elif temp > 80:
                score *= 0.8

        scores[subsystem_type.value] = score

    return scores
```

---

## 5-Phase Progressive Probing Methodology

### Overview

Progressive device exploration with increasing interaction levels. Each phase must complete successfully before advancing.

```
Phase 1: RECONNAISSANCE      → Passive capability reading
Phase 2: PASSIVE_OBSERVATION → Read-only monitoring
Phase 3: CONTROLLED_TESTING  → Isolated single operations
Phase 4: INCREMENTAL_ENABLING → Supervised feature activation
Phase 5: PRODUCTION_INTEGRATION → Full validation
```

### Phase 1: RECONNAISSANCE (Read-Only, No Device Interaction)

**Goal**: Gather information without touching device registers

**Safe Operations**:
1. Read ACPI tables for device descriptions
2. Parse device database entries
3. Check documentation references
4. Identify device group and dependencies
5. Review risk assessment

**Tools**:
```bash
# Using existing probe tool
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-explorer
python3 dsmil_probe.py --phase 1 --device 0x8000

# Manual ACPI inspection
sudo cat /sys/firmware/acpi/tables/DSDT | strings | grep -A 5 "DSMIL0D0"
```

**Expected Output**:
- Device name and ID
- Group membership
- Risk level
- Dependencies
- Known capabilities (from database)

**Decision Point**: If device is quarantined, STOP. Otherwise proceed to Phase 2.

### Phase 2: PASSIVE_OBSERVATION (Read-Only Device Access)

**Goal**: Read device registers without triggering any actions

**Safe Operations**:
1. Read device status register (non-destructive)
2. Read capability flags
3. Read version information
4. Read current configuration (read-only)
5. Monitor for spontaneous activity

**Tools**:
```bash
# Using SMI interface (read-only)
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-devices
python3 dsmil_probe.py --phase 2 --device 0x8000 --read-only

# Manual SMI read
sudo ./smi_read 0x8000  # Reads status without writing
```

**Expected Output**:
- Device status (active/inactive)
- Capability register values
- Version/revision information
- Current state snapshot

**Safety Checks**:
- Monitor system temperature (no increase expected)
- Check kernel logs (dmesg) for errors
- Verify no network activity triggered
- Confirm no disk writes occurred

**Decision Point**: If any unexpected activity detected, STOP and document. Otherwise proceed to Phase 3.

### Phase 3: CONTROLLED_TESTING (Single Safe Operations)

**Goal**: Test individual non-destructive operations in isolation

**Safe Operations** (device-specific):

**For TPM/Crypto Devices (0x8000-0x8002)**:
1. Request RNG bytes (non-destructive)
2. Query algorithm support
3. Read public key
4. Get PCR values
5. Request status update

**For Monitoring Devices (0x8005)**:
1. Request current metrics
2. Get thermal reading
3. Query HSM status
4. Read TPM activity counters

**For Security Devices (0x8010)**:
1. Get intrusion status
2. Read tamper log
3. Query sensor states

**Test Procedure**:
```bash
# Isolated test environment
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-devices

# Test 1: Status query
python3 dsmil_probe.py --device 0x8000 --operation get_status --isolated

# Test 2: Capability query
python3 dsmil_probe.py --device 0x8000 --operation get_capabilities --isolated

# Test 3: Safe function call
python3 dsmil_probe.py --device 0x8000 --operation get_statistics --isolated
```

**Between Each Test**:
1. Wait 30 seconds
2. Check system health
3. Review all logs
4. Verify no side effects
5. Document results

**Safety Checks**:
- Temperature within 2°C of baseline
- No kernel errors or warnings
- No unexpected device state changes
- No network/disk activity
- System responsive

**Decision Point**: If 3 operations succeed without issues, proceed to Phase 4. If any operation fails or causes unexpected behavior, STOP and analyze.

### Phase 4: INCREMENTAL_ENABLING (Supervised Activation)

**Goal**: Enable device features one at a time under supervision

**WARNING**: Only proceed with devices confirmed safe in Phase 3

**Procedure**:
1. Enable single feature
2. Monitor for 5 minutes
3. Test feature operation
4. Disable feature
5. Wait 2 minutes
6. Verify system returns to baseline
7. Document behavior

**Example - Monitoring Device (0x8005)**:
```bash
# Enable monitoring
python3 dsmil_probe.py --device 0x8005 --enable --feature monitoring --supervised

# Monitor for 5 minutes
for i in {1..10}; do
    sleep 30
    python3 dsmil_probe.py --device 0x8005 --get-status
    echo "Check $i/10 complete"
done

# Disable
python3 dsmil_probe.py --device 0x8005 --disable --feature monitoring

# Verify return to baseline
sleep 120
python3 dsmil_probe.py --device 0x8005 --get-status
```

**Features to Test** (in order of increasing risk):
1. Status reporting (lowest risk)
2. Passive monitoring
3. Alert generation
4. Active queries
5. Configuration changes (highest risk in this phase)

**Safety Protocol**:
- Human supervision required for entire phase
- Emergency stop procedure ready
- Full system backup available
- Rollback plan documented
- Network isolated (if applicable)

**Decision Point**: If all features work correctly and disable cleanly, document as "verified safe for supervised use". Consider for Phase 5.

### Phase 5: PRODUCTION_INTEGRATION (Full Deployment)

**Goal**: Integrate device into production dashboard and API

**Prerequisites**:
- ✅ All Phase 1-4 tests passed
- ✅ Device behavior documented
- ✅ Safety procedures verified
- ✅ Rollback tested successfully
- ✅ No quarantine triggers identified

**Integration Steps**:

**1. Update Device Database**
```python
# In dsmil_device_database.py, add to SAFE_DEVICES
SAFE_DEVICES.append(0x8050)  # Example: FDE Controller

# Update device entry with verified functions
ALL_DEVICES[0x8050] = DSMILDevice(
    device_id=0x8050,
    name="Full Disk Encryption Controller",
    group=DeviceGroup.GROUP_4_STORAGE_CONTROL,
    status=DeviceStatus.SAFE,  # Updated from RISKY
    description="Full disk encryption (FDE) and self-encrypting drive management",
    verified_functions=[
        "get_status",
        "get_capabilities",
        "list_volumes",
        "get_encryption_status"
    ],
    safe_operations=[
        "read_status",
        "query_capabilities",
        "list_encrypted_volumes"
    ],
    risk_operations=[
        "enable_encryption",
        "disable_encryption",
        "change_key"
    ]
)
```

**2. Add to Subsystem Controller**
```python
# In dsmil_subsystem_controller.py

def get_fde_status(self):
    """Get Full Disk Encryption status"""
    device = self.devices.get(0x8050)
    if not device:
        return None

    try:
        # Call verified safe operation
        result = self._call_device_operation(0x8050, 'get_status')
        return {
            'operational': True,
            'encrypted_volumes': result.get('volumes', []),
            'encryption_active': result.get('active', False),
            'last_check': time.time()
        }
    except Exception as e:
        return {
            'operational': False,
            'error': str(e)
        }
```

**3. Add Dashboard Integration**
```python
# In ai_gui_dashboard.py, add endpoint

@app.route('/api/dsmil/fde-status')
@require_api_key
def api_fde_status():
    """Get FDE controller status"""
    try:
        status = dsmil_controller.get_fde_status()
        return jsonify({
            'success': True,
            'fde_status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

**4. Add Tests**
```python
# In test_dsmil_api.py or ai_benchmarking.py

def test_fde_controller():
    """Test FDE controller integration"""
    # Test status query
    response = requests.get('http://localhost:5050/api/dsmil/fde-status')
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True

    # Verify expected fields
    assert 'fde_status' in data
    assert 'operational' in data['fde_status']

    print("✓ FDE controller test passed")
```

**5. Update Documentation**
```bash
# Update DSMIL_CURRENT_REFERENCE.md with new device details
# Add to safe devices list
# Document verified functions
# Include API endpoints
```

**6. Commit and Deploy**
```bash
git add -A
git commit -m "feat: Add FDE controller (0x8050) to production

Verified through 5-phase probing methodology:
- Phase 1: RECONNAISSANCE - Device identified and documented
- Phase 2: PASSIVE_OBSERVATION - Read operations successful
- Phase 3: CONTROLLED_TESTING - All safe operations verified
- Phase 4: INCREMENTAL_ENABLING - Feature enable/disable tested
- Phase 5: PRODUCTION_INTEGRATION - Fully integrated

Safe Operations Verified:
- get_status
- get_capabilities
- list_volumes
- get_encryption_status

Dashboard endpoint: /api/dsmil/fde-status
Tests: All passing"

git push
```

---

## Device-Specific Probing Procedures

### Group 0: Core Security (Devices 0x8000-0x800B)

**Verified Safe (Already Integrated)**:
- ✅ 0x8000: Master Controller
- ✅ 0x8005: Audit Logger
- ✅ 0x8008: TPM Interface

**To Probe**:
- 0x8001: Cryptographic Engine
- 0x8002: Secure Key Storage
- 0x8003: Authentication Module
- 0x8004: Access Control
- 0x8006: Integrity Monitor
- 0x8007: Secure Boot Controller

**Quarantined (DO NOT PROBE)**:
- ❌ 0x8009: Emergency Wipe
- ❌ 0x800A: Recovery Controller
- ❌ 0x800B: Hidden Memory Controller

**Probing Order**: 0x8001 → 0x8002 → 0x8003 → 0x8004 → 0x8006 → 0x8007

**Special Considerations**:
- Group 0 controls all other groups - proceed with extreme caution
- Test crypto devices (0x8001, 0x8002) with read-only ops first
- Authentication device (0x8003) may affect system access
- DO NOT test boot controller (0x8007) without full backup

### Group 1: Extended Security (Devices 0x8010-0x801B)

**Verified Safe**:
- ✅ 0x8010: Group 1 Controller

**To Probe**:
- 0x8011: Threat Detection
- 0x8012: Intrusion Prevention
- 0x8013: Network Security
- 0x8014: Malware Scanner
- 0x8015: Behavioral Analysis
- 0x8016: Security Policy Engine
- 0x8017: Incident Response
- 0x8018: Forensics Module
- 0x801A: Vulnerability Scanner
- 0x801B: Security Analytics

**Quarantined**:
- ❌ 0x8019: Network Kill Switch

**Probing Order**: 0x8011 → 0x8013 → 0x801A → 0x8018 → 0x801B → 0x8014 → 0x8015 → 0x8012 → 0x8016 → 0x8017

**Special Considerations**:
- May trigger security alerts - have credentials ready
- Intrusion prevention (0x8012) could block network access
- Test on isolated network if possible
- Have bypass procedure ready

### Group 2: Network & Communications (Devices 0x8020-0x802B)

**Verified Safe**:
- ✅ 0x8020: Network Controller

**To Probe**: All others (0x8021-0x8028, 0x802A, 0x802B)

**Quarantined**:
- ❌ 0x8029: Communications Blackout

**Probing Order**: 0x8027 (Monitor) → 0x8028 (DNS) → 0x8021 (Ethernet) → 0x8022 (WiFi) → 0x8023 (Bluetooth) → others

**Special Considerations**:
- Network devices may affect connectivity
- Have backup internet connection available
- Test WiFi controller may disconnect current connection
- VPN device (0x8024) may route traffic unexpectedly

### Groups 3-6: Lower Priority

**Group 3: Data Processing** (0x8030-0x803B)
- Verified: 0x8030 (Controller)
- Priority: MEDIUM
- Risk: LOW-MEDIUM

**Group 4: Storage Control** (0x8040-0x804B)
- Verified: None
- Priority: MEDIUM-HIGH
- Risk: MEDIUM (could affect disk access)

**Group 5: Peripheral Management** (0x8050-0x805B)
- Verified: None
- Priority: LOW
- Risk: LOW

**Group 6: Training Functions** (0x8060-0x806B)
- Verified: None
- Priority: LOW
- Risk: VERY LOW (training mode)

---

## Safety Protocols

### Before Probing Any Device

**Checklist**:
- [ ] Device NOT in QUARANTINED_DEVICES list
- [ ] Full system backup available
- [ ] Bootable recovery USB ready
- [ ] Network isolated (for network devices)
- [ ] Monitoring tools active (temperature, logs, network)
- [ ] Emergency stop procedure documented
- [ ] Rollback plan written
- [ ] Probe script tested on known-safe device first

### During Probing

**Continuous Monitoring**:
```bash
# Terminal 1: Temperature monitoring
watch -n 1 'sensors | grep Core'

# Terminal 2: Kernel logs
sudo dmesg -w

# Terminal 3: System logs
sudo journalctl -f

# Terminal 4: Network activity
sudo tcpdump -i any -n

# Terminal 5: Probe operations
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-explorer
python3 dsmil_probe.py --device 0x8XXX --phase N
```

**Stop Immediately If**:
- Temperature increases >5°C
- Kernel errors appear
- Unexpected network activity
- System becomes unresponsive
- Any device in different group activates
- Disk I/O spikes unexpectedly

### After Probing

**Verification**:
1. Wait 5 minutes in idle state
2. Check all monitoring outputs
3. Verify system returns to baseline
4. Review all logs
5. Document any anomalies
6. Update device database with findings

**Documentation Template**:
```markdown
## Device 0x8XXX Probing Results

Date: YYYY-MM-DD
Phase Completed: N
Duration: X minutes

### Observations:
- Status: [active/inactive]
- Capabilities: [list discovered capabilities]
- Unexpected Behavior: [none/describe]

### Safety Checks:
- Temperature: [baseline / +X°C]
- Kernel Errors: [none/count]
- Network Activity: [none/describe]
- System Stability: [stable/unstable]

### Conclusion:
[SAFE / RISKY / QUARANTINE]

### Recommendations:
[Next steps or integration plan]
```

---

## Recommended Probing Schedule

### Week 1: Complete Group 0 (Non-Quarantined)
- Day 1: 0x8001 (Crypto Engine) - Phases 1-2
- Day 2: 0x8001 - Phases 3-5
- Day 3: 0x8002 (Key Storage) - Phases 1-3
- Day 4: 0x8002 - Phases 4-5, 0x8003 start
- Day 5: 0x8003, 0x8004 (Authentication & Access Control)

### Week 2: Complete Group 1
- Focus on passive security devices first (monitoring, analytics)
- Leave active devices (intrusion prevention, incident response) for end of week

### Week 3: Complete Group 2
- Start with monitoring/read-only devices
- Test network devices on isolated network
- Verify rollback before production

### Week 4: Groups 3-6 (Lower Priority)
- Batch process lower-risk devices
- Focus on Group 4 (Storage) for production value

---

## API Integration Examples

### Add New Device Endpoint

```python
# In ai_gui_dashboard.py

@app.route('/api/dsmil/device/<device_id>/probe')
@require_api_key
def api_probe_device(device_id):
    """Probe device and return capabilities"""
    try:
        device_id_int = int(device_id, 16) if device_id.startswith('0x') else int(device_id)

        # Safety check
        if device_id_int in QUARANTINED_DEVICES:
            return jsonify({
                'success': False,
                'error': 'Device is quarantined and cannot be probed'
            }), 403

        # Probe device
        result = dsmil_controller.probe_device_safe(device_id_int)

        return jsonify({
            'success': True,
            'device_id': f"0x{device_id_int:04X}",
            'capabilities': result.get('capabilities', {}),
            'status': result.get('status', 'unknown'),
            'risk_level': result.get('risk_level', 'unknown')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### Add Batch Probing Endpoint

```python
@app.route('/api/dsmil/probe-group/<int:group_id>')
@require_api_key
def api_probe_group(group_id):
    """Probe all non-quarantined devices in a group"""
    try:
        if group_id < 0 or group_id > 6:
            return jsonify({'success': False, 'error': 'Invalid group ID'}), 400

        results = []
        device_range = range(0x8000 + (group_id * 0x10), 0x8000 + (group_id * 0x10) + 0x0C)

        for device_id in device_range:
            if device_id not in QUARANTINED_DEVICES:
                result = dsmil_controller.probe_device_safe(device_id)
                results.append({
                    'device_id': f"0x{device_id:04X}",
                    'success': result.get('success', False),
                    'status': result.get('status', 'unknown')
                })

        return jsonify({
            'success': True,
            'group_id': group_id,
            'devices_probed': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

---

## References

**Code Files**:
- `/02-ai-engine/dsmil_subsystem_controller.py` - Main controller
- `/02-ai-engine/dsmil_device_database.py` - Device database
- `/02-tools/dsmil-explorer/dsmil_probe.py` - Automated probe tool
- `/02-tools/dsmil-devices/dsmil_probe.py` - Functional probe tool

**Documentation**:
- `DSMIL_CURRENT_REFERENCE.md` - Current system reference
- `DSMIL_INTEGRATION_SUCCESS.md` - Integration status
- `EXECUTIVE_SUMMARY.md` - Discovery story

**Tools**:
```bash
# Probe automation
/02-tools/dsmil-explorer/dsmil_probe.py

# Manual testing
/02-tools/dsmil-devices/dsmil_probe.py

# Dashboard integration
./start-dashboard.sh
# Access: http://localhost:5050
```

---

**Last Updated**: 2025-11-07
**Next Review**: After Group 0 completion
**Maintained By**: DSMIL Integration Framework
