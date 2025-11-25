# Military Device Interface Testing - Usage Instructions
**TESTBED/DEBUGGER/QADIRECTOR Team**

## Quick Start

### 1. Fix Library Loading Issues
```bash
# Run the library path fix script
./fix_library_path.sh

# For device file creation (requires sudo):
sudo ./fix_library_path.sh
```

### 2. Run Comprehensive Testing
```bash
# Run the complete test suite
./test_military_devices.sh

# View test results
cat testing/reports/test_report_*.json
cat PHASE1_TESTING_COMPLETE_REPORT.md
```

### 3. Manual Library Testing
```bash
# Test library directly
LD_LIBRARY_PATH=./obj ./obj/test_military_interface -h

# Run specific tests
LD_LIBRARY_PATH=./obj ./obj/test_military_interface -v     # Basic tests
LD_LIBRARY_PATH=./obj ./obj/test_military_interface -a     # All devices
LD_LIBRARY_PATH=./obj ./obj/test_military_interface -p     # Performance
LD_LIBRARY_PATH=./obj ./obj/test_military_interface -m     # Monitoring
```

## Test Suite Features

### Safety Validations
- ✅ Thermal safety monitoring (100°C limit)
- ✅ Quarantine enforcement (5 critical devices: 0x8009, 0x800A, 0x800B, 0x8019, 0x8029)  
- ✅ Emergency shutdown mechanisms
- ✅ READ-ONLY operation compliance

### Library Testing
- ✅ Shared library compilation and loading
- ✅ Dependency resolution
- ✅ Test executable functionality
- ✅ Performance characteristics

### Generated Reports
- **JSON Report**: `testing/reports/test_report_YYYYMMDD_HHMMSS.json`
- **Detailed Log**: `testing/logs/military_device_test_YYYYMMDD_HHMMSS.log`
- **Comprehensive Report**: `PHASE1_TESTING_COMPLETE_REPORT.md`

## Critical Issue Status

### ❌ Known Issue: Kernel Module Device Registration
- **Problem**: Module loads but doesn't create `/dev/dsmil-72dev`
- **Impact**: Limited device discovery functionality
- **Safety**: Thermal and quarantine systems still operational
- **Next Steps**: Kernel module debugging required

## MONITOR Agent Integration

```bash
# Start monitoring for real-time feedback
./monitoring/start_monitoring_session.sh

# Check monitoring status
ls monitoring/logs/
```

## File Locations

```
/home/john/LAT5150DRVMIL/
├── fix_library_path.sh              # Library fix script
├── test_military_devices.sh         # Comprehensive test suite  
├── obj/
│   ├── libmilitary_device.so        # Shared library (26KB)
│   └── test_military_interface      # Test executable (26KB)
├── testing/
│   ├── logs/                        # Test logs
│   └── reports/                     # JSON test reports
├── PHASE1_TESTING_COMPLETE_REPORT.md # Final test report
└── TESTING_USAGE_INSTRUCTIONS.md    # This file
```

## Test Results Summary

- **Total Tests**: 22
- **Passed**: 15 (68.1%)
- **Failed**: 1 (device registration)
- **Warnings**: 3 (non-critical)
- **Safety Systems**: ✅ ALL OPERATIONAL