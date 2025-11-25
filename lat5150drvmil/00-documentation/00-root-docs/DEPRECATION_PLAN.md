# DSMIL Platform Deprecation Plan

## Overview

This document outlines the deprecation strategy for legacy DSMIL components after the integration of the new 104-device driver and unified control centre.

**Date:** 2025-11-13
**Version:** 2.0.0
**Status:** Active Deprecation

---

## Deprecated Components

### 1. Legacy Driver (84-device architecture)

**Component:** `01-source/kernel/core/dsmil-84dev.ko`

**Status:** ⚠️ DEPRECATED - Use `dsmil-104dev.ko`

**Reason:**
- Replaced by 104-device architecture
- Has include path issues requiring manual fixes
- Limited to 84 devices (vs 104)
- Less comprehensive token support

**Migration Path:**
```bash
# Old (deprecated)
sudo insmod 01-source/kernel/dsmil-84dev.ko

# New (recommended)
sudo insmod 01-source/kernel/dsmil-104dev.ko
```

**Removal Timeline:** Version 3.0.0 (2026 Q2)

---

### 2. Legacy Control Centres

#### 2.1 Original DSMIL Subsystem Controller

**Component:** `02-ai-engine/dsmil_subsystem_controller.py`

**Status:** ⚠️ DEPRECATED - Use `dsmil_control_centre_104.py`

**Reason:**
- Designed for 84-device architecture
- No integration with new driver IOCTL interface
- Limited to legacy token ranges

**Migration Path:**
```bash
# Old (deprecated)
python3 02-ai-engine/dsmil_subsystem_controller.py

# New (recommended)
python3 02-ai-engine/dsmil_control_centre_104.py
# OR
python3 dsmil.py control
```

#### 2.2 Operation Monitor

**Component:** `02-ai-engine/dsmil_operation_monitor.py`

**Status:** ⚠️ DEPRECATED - Use control centre monitoring mode

**Reason:**
- Monitoring functionality integrated into control centre
- Real-time monitoring via driver interface

**Migration Path:**
```bash
# Old (deprecated)
python3 02-ai-engine/dsmil_operation_monitor.py

# New (recommended)
python3 dsmil.py control
# Then select: Option 4 - System Monitoring
```

#### 2.3 Guided Activation

**Component:** `02-ai-engine/dsmil_guided_activation.py`

**Status:** ⚠️ DEPRECATED - Use control centre activation mode

**Reason:**
- Activation functionality integrated into control centre
- Better safety enforcement
- Integration with cascading discovery

**Migration Path:**
```bash
# Old (deprecated)
python3 02-ai-engine/dsmil_guided_activation.py

# New (recommended)
python3 dsmil.py control
# Then select: Option 2 - Activate Safe Devices
```

---

### 3. Legacy Discovery Scripts

#### 3.1 Basic Discovery

**Component:** `02-tools/dsmil-devices/dsmil_discover.py`

**Status:** ⚠️ DEPRECATED - Use integration adapter

**Reason:**
- Limited discovery capabilities
- No integration with new driver
- Replaced by 4-phase cascading discovery

**Migration Path:**
```python
# Old (deprecated)
from dsmil_discover import discover_devices
devices = discover_devices()

# New (recommended)
from dsmil_integration_adapter import DSMILIntegrationAdapter
adapter = DSMILIntegrationAdapter()
devices = adapter.discover_all_devices_cascading()
```

#### 3.2 Auto Discovery

**Component:** `02-tools/dsmil-devices/dsmil_auto_discover.py`

**Status:** ⚠️ DEPRECATED - Use integration adapter

**Reason:**
- Limited to 84 devices
- No integration with driver interface

**Migration Path:**
```python
# Old (deprecated)
from dsmil_auto_discover import auto_register_all_devices
registry, stats = auto_register_all_devices()

# New (recommended)
from dsmil_integration_adapter import quick_discover
adapter = quick_discover()
```

---

### 4. Legacy Activation Scripts

**Component:** `02-ai-engine/dsmil_device_activation.py`

**Status:** ⚠️ DEPRECATED - Use integration adapter

**Reason:**
- No integration with new driver IOCTL interface
- Limited safety enforcement
- Replaced by unified activation system

**Migration Path:**
```python
# Old (deprecated)
from dsmil_device_activation import DSMILDeviceActivator
activator = DSMILDeviceActivator()
result = activator.activate_device(device_id)

# New (recommended)
from dsmil_integration_adapter import DSMILIntegrationAdapter
adapter = DSMILIntegrationAdapter()
success = adapter.activate_device(device_id)
```

---

### 5. Legacy Database

**Component:** `02-ai-engine/dsmil_device_database.py`

**Status:** ⚠️ DEPRECATED - Use extended database

**Reason:**
- Limited to 84 devices
- Missing new device groups (7 & 8)
- No expansion slots

**Migration Path:**
```python
# Old (deprecated)
from dsmil_device_database import ALL_DEVICES, get_device

# New (recommended)
from dsmil_device_database_extended import (
    ALL_DEVICES_EXTENDED,
    get_device_extended
)
```

---

## Active Components (Current)

### Driver

✅ **`01-source/kernel/core/dsmil-104dev.ko`** - Production driver
- 104 devices
- 3 BIOS systems
- TPM 2.0 authentication
- Full IOCTL interface
- Clean build (no include path issues)

### Control Centre

✅ **`dsmil.py`** - Unified entry point
- Single command interface
- Driver management
- Control centre launcher
- Diagnostics

✅ **`02-ai-engine/dsmil_control_centre_104.py`** - Main UI
- Interactive menu
- Discovery, activation, monitoring
- Comprehensive reporting

### Integration Layer

✅ **`02-ai-engine/dsmil_integration_adapter.py`** - Unified API
- 4-phase cascading discovery
- Multi-method activation
- System monitoring
- Legacy compatibility

✅ **`02-ai-engine/dsmil_driver_interface.py`** - IOCTL bindings
- All 12 IOCTL commands
- Token operations
- TPM authentication
- BIOS management

✅ **`02-ai-engine/dsmil_device_database_extended.py`** - 104-device database
- All 104 devices
- 9 device groups
- 20 expansion slots

---

## Migration Timeline

### Phase 1: Deprecation Announcement (Current)
- **Date:** 2025-11-13
- **Status:** ✅ Complete
- All legacy components marked as deprecated
- Documentation updated
- Migration paths provided

### Phase 2: Parallel Support (2025 Q4 - 2026 Q1)
- **Duration:** 6 months
- Both old and new systems available
- Users encouraged to migrate
- Legacy components maintained for critical bugs only

### Phase 3: Deprecation Warnings (2026 Q2)
- **Date:** 2026-04-01
- Legacy components emit deprecation warnings
- Documentation updated with removal date
- Final migration push

### Phase 4: Removal (2026 Q3)
- **Date:** 2026-07-01
- Legacy components moved to `_archived/` directory
- Only new system supported
- Legacy code available for reference but not maintained

---

## Automated Migration

### Script: `migrate_to_v2.sh`

```bash
#!/bin/bash
# Automatic migration from legacy to v2.0 system

echo "DSMIL Platform Migration: v1.x → v2.0"
echo "======================================"

# 1. Update imports
echo "[1/4] Updating Python imports..."
find . -name "*.py" -type f -exec sed -i \
    's/from dsmil_device_database import/from dsmil_device_database_extended import/g' {} +
find . -name "*.py" -type f -exec sed -i \
    's/ALL_DEVICES/ALL_DEVICES_EXTENDED/g' {} +

# 2. Update control centre calls
echo "[2/4] Updating control centre references..."
find . -name "*.py" -type f -exec sed -i \
    's/dsmil_subsystem_controller/dsmil_control_centre_104/g' {} +

# 3. Update driver references
echo "[3/4] Updating driver references..."
find . -name "*.sh" -type f -exec sed -i \
    's/dsmil-84dev/dsmil-104dev/g' {} +

# 4. Create backup
echo "[4/4] Creating backup of changed files..."
mkdir -p _migration_backup
cp -r 02-ai-engine/_migration_backup/ 2>/dev/null || true

echo ""
echo "Migration complete!"
echo ""
echo "Next steps:"
echo "  1. Review changed files"
echo "  2. Test with: python3 dsmil.py diagnostics"
echo "  3. Run: python3 dsmil.py build && sudo python3 dsmil.py load"
```

---

## Breaking Changes

### API Changes

#### 1. Discovery Function Signature

**Old:**
```python
devices = discover_devices()  # Returns list of token IDs
```

**New:**
```python
adapter = DSMILIntegrationAdapter()
devices = adapter.discover_all_devices_cascading()  # Returns list of device IDs (0-103)
```

**Note:** Old API returned token IDs (0x8000+), new API returns device IDs (0-103)

#### 2. Activation Return Values

**Old:**
```python
result = activator.activate_device(device_id)
# Returns: ActivationResult object
```

**New:**
```python
success = adapter.activate_device(device_id)
# Returns: boolean
```

#### 3. Database Functions

**Old:**
```python
device = get_device(device_id)  # 84 devices max
```

**New:**
```python
device = get_device_extended(device_id)  # 104 devices
```

---

## Backwards Compatibility

### Compatibility Layer

For gradual migration, a compatibility shim is available:

**Component:** `02-ai-engine/dsmil_legacy_compat.py`

```python
"""Legacy compatibility layer for v1.x code"""

from dsmil_device_database_extended import *
from dsmil_integration_adapter import DSMILIntegrationAdapter

# Legacy function names
def discover_devices():
    """Legacy discovery function"""
    adapter = DSMILIntegrationAdapter()
    device_ids = adapter.discover_all_devices_cascading()
    # Convert device IDs to token IDs for compatibility
    return [0x8000 + (d * 3) for d in device_ids]

def get_device(device_id):
    """Legacy get_device function"""
    return get_device_extended(device_id)

# Maintain old names
ALL_DEVICES = ALL_DEVICES_EXTENDED
SAFE_DEVICES = SAFE_DEVICES_EXTENDED
QUARANTINED_DEVICES = QUARANTINED_DEVICES_EXTENDED
```

**Usage:**
```python
# Import compatibility layer at top of legacy scripts
from dsmil_legacy_compat import *
# Rest of code works unchanged
```

---

## Support Policy

### Version 2.0+ (Current)
- ✅ Full support
- ✅ Bug fixes
- ✅ New features
- ✅ Documentation updates

### Version 1.x (Legacy)
- ⚠️ Critical bugs only (until 2026 Q2)
- ❌ No new features
- ⚠️ Security fixes only
- ❌ No documentation updates

### Version 0.x (Ancient)
- ❌ No support
- ❌ Archived only

---

## FAQ

### Q: Can I still use the 84-device driver?

**A:** Yes, but it's deprecated. The 84-device driver (`dsmil-84dev.ko`) will continue to work but:
- Has known include path issues
- Limited device support (84 vs 104)
- No new features
- Will be removed in v3.0.0 (2026 Q2)

**Recommended:** Migrate to `dsmil-104dev.ko`

### Q: Will my existing scripts break?

**A:** Not immediately. The legacy components remain available during the deprecation period (6 months). However:
- Update scripts to use new API
- Use compatibility layer for gradual migration
- Test thoroughly before production deployment

### Q: How do I know which version I'm using?

**A:** Check the driver version:
```bash
# New driver
cat /sys/class/dsmil/dsmil0/driver_version
# Output: 5.2.0 (104 devices)

# Old driver
cat /sys/class/dsmil/dsmil0/driver_version
# Output: 4.x or earlier (84 devices)
```

### Q: What if I find a bug in legacy components?

**A:**
- Critical security bugs: Will be fixed
- Critical functionality bugs: May be fixed
- Minor bugs: Won't be fixed, migrate to v2.0
- Enhancement requests: Won't be implemented

---

## Resources

### Documentation
- **New System:** `02-ai-engine/README_INTEGRATION.md`
- **Driver API:** `01-source/kernel/API_REFERENCE.md`
- **Migration:** This document

### Support
- **Issues:** Report in project issue tracker
- **Questions:** See documentation first
- **Migration Help:** Run `python3 dsmil.py diagnostics`

---

**Last Updated:** 2025-11-13
**Status:** Active
**Next Review:** 2026-01-01
