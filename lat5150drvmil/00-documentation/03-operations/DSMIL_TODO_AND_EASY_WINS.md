# DSMIL Framework TODO List & Easy Wins
**Last Updated**: 2025-11-07
**Purpose**: Quick reference for pending improvements and easy wins

---

## 🎯 Easy Wins (Implement These First)

### Priority 1: High Value, Low Effort (< 2 hours each)

#### ✅ Win #1: Enhanced Thermal Monitoring
**Time**: 1-2 hours
**Value**: Immediate safety improvement
**Risk**: None (read-only)
**File**: `02-ai-engine/dsmil_subsystem_controller.py`

**Implementation**:
```python
def get_thermal_status_enhanced(self):
    """Enhanced thermal monitoring with per-core readings"""
    from pathlib import Path

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

    return {
        'zones': temps,
        'max_temp': max([t['temp_c'] for t in temps]) if temps else 0,
        'overall_status': 'critical' if any(t['status'] == 'critical' for t in temps) else
                         'warning' if any(t['status'] == 'warning' for t in temps) else 'normal'
    }
```

**Integration**:
- Add to `_check_thermal()` in subsystem controller
- Expose via `/api/dsmil/thermal-enhanced` endpoint
- Add to dashboard thermal display

---

#### ✅ Win #2: TPM PCR State Tracking
**Time**: 1-2 hours
**Value**: Better attestation visibility
**Risk**: None (read-only)
**File**: `02-ai-engine/dsmil_subsystem_controller.py`

**Implementation**:
```python
def get_tpm_pcr_state(self):
    """Get current TPM PCR measurements"""
    import subprocess

    try:
        result = subprocess.run(
            ['tpm2_pcrread', 'sha256:0,1,2,3,4,5,6,7,8'],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            pcrs = {}
            for line in result.stdout.split('\n'):
                if ':' in line and 'sha256' not in line.lower():
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        pcr_num = parts[0].strip()
                        pcr_val = parts[1].strip()
                        pcrs[pcr_num] = pcr_val

            return {
                'success': True,
                'pcrs': pcrs,
                'timestamp': time.time()
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}

    return {'success': False, 'error': 'TPM read failed'}

def get_tpm_event_log(self):
    """Read TPM event log for measurement count"""
    from pathlib import Path

    event_log_path = Path('/sys/kernel/security/tpm0/binary_bios_measurements')
    if event_log_path.exists():
        try:
            event_data = event_log_path.read_bytes()
            # Count events (simplified - each event starts with specific header)
            event_count = event_data.count(b'\x00\x00\x00')  # Simplified count

            return {
                'event_count': event_count,
                'log_size_bytes': len(event_data),
                'log_available': True
            }
        except Exception as e:
            return {'event_count': 0, 'error': str(e), 'log_available': False}

    return {'event_count': 0, 'log_available': False}
```

**Integration**:
- Add to `_check_tpm_attestation()` subsystem check
- Update `/api/tpm/status` to include PCR state
- Add TPM event count to dashboard

---

#### ✅ Win #3: Device Status Caching
**Time**: 30 minutes
**Value**: Reduce SMI calls, improve performance
**Risk**: None
**File**: `02-ai-engine/dsmil_subsystem_controller.py`

**Implementation**:
```python
from datetime import datetime, timedelta

class DSMILSubsystemController:
    def __init__(self):
        # ... existing init ...
        self.device_status_cache = {}
        self.cache_ttl_seconds = 5  # 5 second cache

    def get_device_status_cached(self, device_id: int):
        """Get device status with caching to reduce SMI calls"""
        cache_key = f"status_{device_id:04X}"
        now = datetime.now()

        # Check cache
        if cache_key in self.device_status_cache:
            cached_time, cached_value = self.device_status_cache[cache_key]
            if (now - cached_time).total_seconds() < self.cache_ttl_seconds:
                return cached_value

        # Cache miss or expired - fetch fresh
        status = self._read_device_status(device_id)  # Your existing method
        self.device_status_cache[cache_key] = (now, status)
        return status

    def clear_device_cache(self, device_id: Optional[int] = None):
        """Clear cache for specific device or all devices"""
        if device_id:
            cache_key = f"status_{device_id:04X}"
            self.device_status_cache.pop(cache_key, None)
        else:
            self.device_status_cache.clear()
```

**Usage**:
- Replace direct status reads with cached version in loops
- Clear cache after device operations that change state
- Adjust `cache_ttl_seconds` based on needs (5s default)

---

#### ✅ Win #4: Operation History Logging
**Time**: 1 hour
**Value**: Debugging, audit trail
**Risk**: None
**File**: `02-ai-engine/dsmil_subsystem_controller.py`

**Implementation**:
```python
from collections import deque
from datetime import datetime

class DSMILSubsystemController:
    def __init__(self):
        # ... existing init ...
        self.operation_history = deque(maxlen=1000)  # Last 1000 operations

    def log_operation(self, device_id: int, operation: str,
                     success: bool, details: str = "", value: Optional[int] = None):
        """Log device operation for history and debugging"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': f"0x{device_id:04X}",
            'device_name': self.devices.get(device_id, {}).get('name', 'Unknown'),
            'operation': operation,
            'success': success,
            'details': details,
            'value': value
        }
        self.operation_history.append(entry)

    def get_operation_history(self, device_id: Optional[int] = None,
                             limit: int = 100,
                             operation_type: Optional[str] = None):
        """Get operation history with optional filtering"""
        history = list(self.operation_history)

        # Filter by device
        if device_id:
            history = [h for h in history if h['device_id'] == f"0x{device_id:04X}"]

        # Filter by operation type
        if operation_type:
            history = [h for h in history if h['operation'] == operation_type]

        return history[-limit:]

    def get_operation_stats(self):
        """Get statistics about operations"""
        if not self.operation_history:
            return {}

        total = len(self.operation_history)
        success = sum(1 for op in self.operation_history if op['success'])

        # Count by device
        by_device = {}
        for op in self.operation_history:
            dev = op['device_id']
            by_device[dev] = by_device.get(dev, 0) + 1

        return {
            'total_operations': total,
            'successful': success,
            'failed': total - success,
            'success_rate': (success / total) * 100 if total > 0 else 0,
            'operations_by_device': by_device,
            'most_active_device': max(by_device.items(), key=lambda x: x[1])[0] if by_device else None
        }
```

**Integration**:
- Call `log_operation()` after every device operation
- Add endpoint: `/api/dsmil/operation-history`
- Add endpoint: `/api/dsmil/operation-stats`
- Display in dashboard

---

#### ✅ Win #5: Subsystem Health Scores
**Time**: 30 minutes
**Value**: Quick health overview
**Risk**: None
**File**: `02-ai-engine/dsmil_subsystem_controller.py`

**Implementation**:
```python
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

        # Subsystem-specific adjustments
        if subsystem_type == SubsystemType.THERMAL:
            temp = status.status_info.get('current_temp', 0)
            if temp > 90:
                score *= 0.3  # Critical thermal
            elif temp > 85:
                score *= 0.6
            elif temp > 80:
                score *= 0.8  # Warning thermal

        elif subsystem_type == SubsystemType.TPM_ATTESTATION:
            tpm_avail = status.status_info.get('tpm_available', False)
            if not tpm_avail:
                score *= 0.5

        elif subsystem_type == SubsystemType.SECURITY:
            quarantined_attempts = status.status_info.get('quarantine_violations', 0)
            if quarantined_attempts > 0:
                score *= 0.7  # Security concerns

        scores[subsystem_type.value] = round(score, 2)

    # Overall system health
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    return {
        'subsystem_scores': scores,
        'overall_health': round(avg_score, 2),
        'status': 'excellent' if avg_score > 0.9 else
                 'good' if avg_score > 0.7 else
                 'fair' if avg_score > 0.5 else
                 'poor'
    }
```

**Integration**:
- Add to dashboard homepage
- Create visual health indicators
- Add endpoint: `/api/dsmil/health-score`

---

### Priority 2: Medium Value, Medium Effort (2-4 hours each)

#### 🔧 Device Activation Implementation
**Time**: 2-4 hours
**Value**: Core functionality
**Risk**: Medium
**File**: `02-ai-engine/dsmil_subsystem_controller.py:550`

**Current Code**:
```python
# TODO: Implement actual device activation via sysfs/ioctl
# For now, return success for safe devices
```

**Implementation Options**:

**Option A: sysfs Interface** (if kernel module exposes it)
```python
def activate_safe_device(self, device_id: int, value: int = 1) -> Tuple[bool, str]:
    """Activate safe device via sysfs"""
    from pathlib import Path

    # Check if device is safe
    if not self.is_device_safe(device_id):
        self.log_operation(device_id, 'activate', False, 'Device not safe')
        return (False, f"Device 0x{device_id:04X} is not safe for activation")

    # Try sysfs path
    sysfs_path = Path(f'/sys/class/dsmil/device_{device_id:04X}/activate')
    if sysfs_path.exists():
        try:
            sysfs_path.write_text(str(value))
            self.log_operation(device_id, 'activate', True, f'Value: {value}', value)
            return (True, f"Device 0x{device_id:04X} activated")
        except Exception as e:
            self.log_operation(device_id, 'activate', False, str(e))
            return (False, f"Activation failed: {e}")

    self.log_operation(device_id, 'activate', False, 'No sysfs interface')
    return (False, "sysfs interface not available")
```

**Option B: ioctl Interface** (if using character device)
```python
import fcntl

# IOCTL magic numbers (from kernel driver)
DSMIL_IOC_MAGIC = 0xD5
DSMIL_IOC_ACTIVATE = 0x01

def activate_safe_device(self, device_id: int, value: int = 1) -> Tuple[bool, str]:
    """Activate safe device via ioctl"""

    if not self.is_device_safe(device_id):
        self.log_operation(device_id, 'activate', False, 'Device not safe')
        return (False, f"Device 0x{device_id:04X} is not safe for activation")

    try:
        # Open device node
        with open('/dev/dsmil0', 'rb+', buffering=0) as dev:
            # Pack activation command
            import struct
            cmd_data = struct.pack('HH', device_id, value)

            # Send ioctl
            fcntl.ioctl(dev, (DSMIL_IOC_MAGIC << 8) | DSMIL_IOC_ACTIVATE, cmd_data)

            self.log_operation(device_id, 'activate', True, f'Value: {value}', value)
            return (True, f"Device 0x{device_id:04X} activated with value {value}")

    except FileNotFoundError:
        self.log_operation(device_id, 'activate', False, 'Device node not found')
        return (False, "/dev/dsmil0 not found - kernel module not loaded?")
    except Exception as e:
        self.log_operation(device_id, 'activate', False, str(e))
        return (False, f"ioctl failed: {e}")
```

**Testing Procedure**:
1. Verify kernel module loaded: `lsmod | grep dsmil`
2. Check for sysfs: `ls /sys/class/dsmil/`
3. Check for device node: `ls -l /dev/dsmil*`
4. Test with safe device 0x8000 first
5. Verify activation with status read
6. Test deactivation/rollback

---

#### 🔧 Audit Framework Storage
**Time**: 1-2 hours
**Value**: Compliance, debugging
**Risk**: Low
**File**: `01-source/kernel/dsmil_audit_framework.c:1064`

**Implementation** (Python layer):
```python
import sqlite3
from datetime import datetime

class DSMILAuditStorage:
    """Persistent storage for audit events"""

    def __init__(self, db_path: str = "/var/lib/dsmil/audit.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for audit storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                device_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                user TEXT,
                success BOOLEAN NOT NULL,
                details TEXT,
                risk_level TEXT,
                INDEX idx_timestamp (timestamp),
                INDEX idx_device (device_id),
                INDEX idx_operation (operation)
            )
        ''')

        conn.commit()
        conn.close()

    def store_event(self, device_id: int, operation: str, success: bool,
                   user: str = None, details: str = "", risk_level: str = "low"):
        """Store audit event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audit_events
            (timestamp, device_id, operation, user, success, details, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().timestamp(),
            f"0x{device_id:04X}",
            operation,
            user,
            success,
            details,
            risk_level
        ))

        conn.commit()
        conn.close()

    def get_events(self, limit: int = 100, device_id: int = None,
                  start_time: float = None, end_time: float = None):
        """Query audit events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        if device_id:
            query += " AND device_id = ?"
            params.append(f"0x{device_id:04X}")

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return events
```

**Integration**:
```python
# In DSMILSubsystemController.__init__():
self.audit_storage = DSMILAuditStorage()

# In every device operation:
self.audit_storage.store_event(device_id, operation, success, details=details)
```

---

## 📋 TODO List from Codebase

### High Priority (Core Functionality)

| TODO | File | Line | Priority | Effort | Status |
|------|------|------|----------|--------|--------|
| Implement device activation via sysfs/ioctl | dsmil_subsystem_controller.py | 550 | HIGH | MEDIUM | 🔴 Pending |
| Implement actual thermal reading | dsmil_hal.c | 727 | HIGH | LOW | 🔴 Pending |
| Track TPM state | dell-millspec-enhanced.c | 1725 | MEDIUM | LOW | 🔴 Pending |
| Track TPM event count | dell-millspec-enhanced.c | 1732 | MEDIUM | LOW | 🔴 Pending |

### Medium Priority (Enhancements)

| TODO | File | Line | Priority | Effort | Status |
|------|------|------|----------|--------|--------|
| Implement event logging with trace infrastructure | dell-millspec-enhanced.c | 235 | MEDIUM | MEDIUM | 🔴 Pending |
| Implement proper authorization checking | dsmil_hal.c | 799 | MEDIUM | MEDIUM | 🔴 Pending |
| Implement audit storage write | dsmil_audit_framework.c | 1064 | MEDIUM | LOW | 🔴 Pending |

### Low Priority (Nice to Have)

| TODO | File | Line | Priority | Effort | Status |
|------|------|------|----------|--------|--------|
| Implement ring buffer reading | dell-millspec-enhanced.c | 1569 | LOW | MEDIUM | 🔴 Pending |
| Implement firmware update | dell-millspec-enhanced.c | 1813 | LOW | HIGH | 🔴 Pending |
| Implement key migration protocols | dell-millspec-enhanced.c | 2566 | LOW | HIGH | 🔴 Pending |
| Add hardware fault detection (Rust) | rust/src/smi.rs | 88 | LOW | LOW | 🔴 Pending |

### Kernel-Level TODOs (Require Kernel Development)

| TODO | File | Description | Priority |
|------|------|-------------|----------|
| dsmil-72dev.c:2291 | Implement device activation via ACPI | MEDIUM |
| dsmil-72dev.c:2310 | Implement device deactivation via ACPI | MEDIUM |
| dsmil_safety.c:344 | Add device state checking logic | LOW |
| dsmil_safety.c:354 | Implement actual thermal monitoring | LOW |

---

## 🚀 Implementation Roadmap

### Phase 1: Easy Wins (Week 1)
- [x] None yet - all pending
- [ ] Implement enhanced thermal monitoring
- [ ] Add TPM PCR state tracking
- [ ] Add device status caching
- [ ] Implement operation history logging
- [ ] Add subsystem health scores

**Expected Outcome**: Improved monitoring, better performance, audit trail

### Phase 2: Core Functionality (Week 2-3)
- [ ] Implement device activation (sysfs/ioctl)
- [ ] Add audit storage persistence
- [ ] Implement authorization checking
- [ ] Complete thermal monitoring integration

**Expected Outcome**: Full device control capability, compliance-ready audit system

### Phase 3: Device Discovery (Week 4-8)
- [ ] Probe all Group 0 non-quarantined devices
- [ ] Probe all Group 1 non-quarantined devices
- [ ] Probe all Group 2 non-quarantined devices
- [ ] Document all discovered functions

**Expected Outcome**: Complete functional map of all 79 safe/risky devices

### Phase 4: Advanced Features (Future)
- [ ] Event logging with trace infrastructure
- [ ] Ring buffer implementation
- [ ] Firmware update capability
- [ ] Key migration protocols

**Expected Outcome**: Production-grade platform with all features

---

## 📊 Progress Tracking

### Current Status
- **Easy Wins Completed**: 0 / 5
- **High Priority TODOs**: 0 / 4 complete
- **Medium Priority TODOs**: 0 / 3 complete
- **Devices Probed**: 6 / 84 (7.1%)
- **Devices Safe for Production**: 6 / 84 (7.1%)

### Metrics to Track
```python
# Add to dashboard
{
    "easy_wins_completed": 0,
    "high_priority_todos": 0,
    "total_todos": 15,
    "devices_probed": 6,
    "devices_safe": 6,
    "devices_quarantined": 5,
    "devices_pending": 73,
    "completion_percentage": 7.1
}
```

---

## 🔗 Quick Links

**Code Locations**:
- Main Controller: `02-ai-engine/dsmil_subsystem_controller.py`
- Device Database: `02-ai-engine/dsmil_device_database.py`
- Probe Tool: `02-tools/dsmil-explorer/dsmil_probe.py`

**Documentation**:
- Probing Guide: `00-documentation/03-operations/DSMIL_DEVICE_PROBING_GUIDE.md`
- Current Reference: `00-documentation/00-root-docs/DSMIL_CURRENT_REFERENCE.md`

**Dashboard**:
```bash
./start-dashboard.sh
# http://localhost:5050
```

---

**Last Updated**: 2025-11-07
**Next Review**: After Easy Wins completion
**Maintained By**: DSMIL Integration Framework
