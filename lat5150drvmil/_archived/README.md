# DSMIL Legacy Components Archive

This directory contains deprecated DSMIL v1.x components that have been archived.

## Archive Schedule

**Status:** Awaiting Phase 4 migration (2026 Q3)

**Current Phase:** Phase 2 - Parallel Support (2025 Q4 - 2026 Q1)

## Components to be Archived in Phase 4 (2026 Q3)

### Legacy Drivers
- `01-source/kernel/core/dsmil-84dev.ko` - 84-device driver

### Legacy Control Centres
- `02-ai-engine/dsmil_subsystem_controller.py` - Original controller
- `02-ai-engine/dsmil_operation_monitor.py` - Operation monitor
- `02-ai-engine/dsmil_guided_activation.py` - Guided activation

### Legacy Discovery & Activation
- `02-tools/dsmil-devices/dsmil_discover.py` - Basic discovery
- `02-tools/dsmil-devices/dsmil_auto_discover.py` - Auto discovery
- `02-ai-engine/dsmil_device_activation.py` - Device activation

### Legacy Database
- `02-ai-engine/dsmil_device_database.py` - 84-device database

## Migration Timeline

### Phase 1: Deprecation Announcement (2025-11-13) ✅
- All legacy components marked as deprecated
- Deprecation warnings added to all files
- Documentation updated
- Migration paths provided

### Phase 2: Parallel Support (2025 Q4 - 2026 Q1) 🟡
- **Current Phase**
- Both old and new systems available
- Users encouraged to migrate
- Legacy components maintained for critical bugs only

### Phase 3: Deprecation Warnings (2026 Q2) 🔵
- Legacy components emit runtime deprecation warnings
- Documentation updated with removal date
- Final migration push

### Phase 4: Archival (2026 Q3) 🔴
- **Target Date:** 2026-07-01
- Legacy components moved to `_archived/` directory
- Only new system supported
- Legacy code available for reference but not maintained

## Archive Structure (Post-2026 Q3)

```
_archived/
├── README.md (this file)
├── v1.x/
│   ├── kernel/
│   │   └── dsmil-84dev.ko
│   ├── control-centres/
│   │   ├── dsmil_subsystem_controller.py
│   │   ├── dsmil_operation_monitor.py
│   │   └── dsmil_guided_activation.py
│   ├── discovery/
│   │   ├── dsmil_discover.py
│   │   └── dsmil_auto_discover.py
│   ├── activation/
│   │   └── dsmil_device_activation.py
│   └── database/
│       └── dsmil_device_database.py
└── ARCHIVAL_LOG.md (created when archived)
```

## Migration Resources

### Documentation
- **Deprecation Plan:** `../DEPRECATION_PLAN.md`
- **Integration Guide:** `../02-ai-engine/README_INTEGRATION.md`
- **API Reference:** `../01-source/kernel/API_REFERENCE.md`

### Migration Tools
- **Automated Migration:** `../migrate_to_v2.sh`
- **Compatibility Layer:** `../02-ai-engine/dsmil_legacy_compat.py` (temporary)
- **Unified Entry Point:** `../dsmil.py`

### New Components (v2.0+)
- **Driver:** `01-source/kernel/core/dsmil-104dev.ko` (104 devices)
- **Control Centre:** `02-ai-engine/dsmil_control_centre_104.py`
- **Integration Adapter:** `02-ai-engine/dsmil_integration_adapter.py`
- **Driver Interface:** `02-ai-engine/dsmil_driver_interface.py`
- **Extended Database:** `02-ai-engine/dsmil_device_database_extended.py`

## Support Policy

### Version 2.0+ (Current)
- ✅ Full support
- ✅ Bug fixes
- ✅ New features
- ✅ Documentation updates

### Version 1.x (Deprecated)
- ⚠️ Critical bugs only (until 2026 Q2)
- ❌ No new features
- ⚠️ Security fixes only
- ❌ No documentation updates

### Archived Components (Post-2026 Q3)
- ❌ No support
- ❌ No maintenance
- 📖 Reference only

## Why Deprecation?

The v2.0 system provides significant improvements:

1. **Extended Capacity:** 104 devices (vs 84)
2. **Clean Architecture:** No include path issues
3. **Direct Driver Access:** IOCTL interface for reliability
4. **Enhanced Safety:** 4-phase cascading discovery
5. **Better Performance:** Optimized token operations
6. **TPM 2.0 Integration:** Hardware-backed authentication
7. **Unified Interface:** Single entry point (`dsmil.py`)
8. **Comprehensive Docs:** 7 detailed guides

## Contact & Support

For migration assistance:
- Run diagnostics: `python3 dsmil.py diagnostics`
- Read migration guide: `DEPRECATION_PLAN.md`
- Review integration docs: `00-documentation/`

---

**Last Updated:** 2025-11-13
**Status:** Archive directory prepared, awaiting Phase 4
**Next Review:** 2026-04-01 (Phase 3)
