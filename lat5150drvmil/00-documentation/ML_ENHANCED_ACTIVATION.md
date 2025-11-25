# ML-Enhanced DSMIL Activation System

## Overview

This document describes the **ML-Enhanced DSMIL Activation System**, a mission-critical component that provides intelligent, automated hardware discovery and device activation with real-time safety monitoring.

## Key Features

### 1. **Automated Hardware Discovery**
- **Multi-Interface Scanning**: Scans SMBIOS tokens, ACPI tables, and sysfs interfaces
- **Dynamic Device Detection**: Discovers DSMIL devices without prior knowledge
- **Hardware Address Mapping**: Identifies device addresses across multiple ranges

### 2. **Machine Learning Classification**
- **Device Capability Prediction**: Uses ML to predict device functions and capabilities
- **Safety Level Classification**: Automatically classifies devices as safe, monitored, caution, or quarantined
- **Confidence Scoring**: Provides confidence levels for ML predictions

### 3. **Intelligent Activation Sequencing**
- **Dependency Resolution**: Automatically determines and respects device dependencies
- **Priority Calculation**: Orders activations based on safety, dependencies, and thermal impact
- **Thermal Impact Prediction**: Estimates thermal load before activation

### 4. **Real-Time Safety Monitoring**
- **Continuous Thermal Monitoring**: Tracks temperature during activation
- **Emergency Halt**: Automatically stops activation if thermal critical threshold is reached
- **Rollback Capability**: Can rollback devices if issues are detected

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              ML-Enhanced Activation System                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐   ┌──────────────────┐                │
│  │  ML Discovery   │   │   Device         │                │
│  │  Engine         │──▶│   Activation     │                │
│  └─────────────────┘   │   Engine         │                │
│          │              └──────────────────┘                │
│          │                       │                           │
│          ▼                       ▼                           │
│  ┌─────────────────┐   ┌──────────────────┐                │
│  │  Device         │   │   Subsystem      │                │
│  │  Database       │   │   Controller     │                │
│  └─────────────────┘   └──────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. `dsmil_ml_discovery.py`

Machine learning-enhanced hardware discovery system.

**Key Classes:**
- `DSMILMLDiscovery`: Main discovery engine
- `HardwareDevice`: Discovered device representation
- `DeviceSafetyLevel`: ML-predicted safety classifications

**Methods:**
- `scan_smbios_tokens()`: Scan SMBIOS for DSMIL tokens
- `scan_acpi_devices()`: Scan ACPI tables for device signatures
- `scan_sysfs_interfaces()`: Scan sysfs for device interfaces
- `predict_device_capabilities()`: ML classification of device function
- `estimate_thermal_impact()`: Predict thermal load
- `discover_all_devices()`: Comprehensive multi-interface scan

**Usage:**
```bash
# Run discovery only
sudo python3 02-ai-engine/dsmil_ml_discovery.py

# Generates report at: /tmp/dsmil_ml_discovery_report.json
```

### 2. `dsmil_integrated_activation.py`

End-to-end activation workflow integrating discovery with activation.

**Key Classes:**
- `DSMILIntegratedActivation`: Main workflow orchestrator
- `WorkflowStage`: Workflow stage tracking
- `WorkflowStatus`: Current workflow status

**Workflow Phases:**
1. **Discovery**: Multi-interface hardware scan
2. **Analysis**: ML classification and sequence planning
3. **Activation**: Intelligent device activation
4. **Monitoring**: Post-activation stability monitoring

**Usage:**
```bash
# Full workflow
sudo python3 02-ai-engine/dsmil_integrated_activation.py

# Discovery only
sudo python3 02-ai-engine/dsmil_integrated_activation.py --no-activation

# Interactive mode (confirm each device)
sudo python3 02-ai-engine/dsmil_integrated_activation.py --interactive

# Custom monitoring duration
sudo python3 02-ai-engine/dsmil_integrated_activation.py --monitor-duration 60

# Generate report
sudo python3 02-ai-engine/dsmil_integrated_activation.py --report /tmp/my_report.json
```

### 3. `launch-ml-enhanced-activation.sh`

Quick launcher for the ML-enhanced system without tmux.

**Usage:**
```bash
sudo ./launch-ml-enhanced-activation.sh
```

**Options:**
1. Full ML-Enhanced Workflow (Discovery + Activation + Monitoring)
2. Discovery Only (No activation)
3. Interactive Mode (Confirm each device)
4. Run with custom monitoring duration

### 4. `launch-dsmil-control-center.sh` (Updated)

Enhanced control center with ML integration.

**Changes:**
- **Fixed log monitor**: Robust error handling (line 61 issue resolved)
- **ML-enhanced activation option**: New menu in Device Activation pane
- **Additional log monitoring**: Tracks ML discovery and integrated activation logs

**Usage:**
```bash
sudo ./launch-dsmil-control-center.sh
```

**Activation Interface Options:**
1. ML-Enhanced Integrated Activation (RECOMMENDED)
2. Guided Manual Activation (Classic)
3. Discovery Only (No Activation)

## Workflow Example

### End-to-End Activation Workflow

```bash
# 1. Launch ML-enhanced system
sudo ./launch-ml-enhanced-activation.sh

# 2. Select mode 1 (Full workflow)

# The system will:
# - Scan SMBIOS tokens for DSMIL devices
# - Scan ACPI tables for device signatures
# - Scan sysfs interfaces for device nodes
# - Use ML to classify each discovered device
# - Calculate optimal activation sequence
# - Activate devices in priority order
# - Monitor thermal conditions continuously
# - Generate comprehensive report

# 3. Review results
cat /tmp/dsmil_integrated_workflow_report.json
```

### Discovery-Only Workflow

```bash
# Run discovery without activation
sudo python3 02-ai-engine/dsmil_integrated_activation.py --no-activation

# Review discovered devices
cat /tmp/dsmil_ml_discovery_report.json
```

## Safety Features

### 1. **Quarantine Protection**
- Known dangerous devices (0x8009, 0x800A, 0x800B, 0x8019, 0x8029) are automatically quarantined
- Quarantined devices CANNOT be activated
- ML system learns quarantine patterns

### 2. **Thermal Protection**
- Continuous temperature monitoring
- Automatic halt if temperature exceeds 90°C
- Warning at 85°C
- Per-device thermal impact estimation

### 3. **Dependency Management**
- Master controller (0x8000) must be active first
- Dependencies automatically resolved
- Activation blocked if dependencies not met

### 4. **Rollback Capability**
- Rollback points created before each activation
- Can revert devices if issues detected
- Preserves system state information

## ML Classification

### Device Categories
- **security**: Cryptographic, authentication, TPM devices
- **network**: Ethernet, WiFi, Bluetooth, VPN devices
- **storage**: Disk, RAID, NVMe devices
- **thermal**: Cooling, temperature monitoring
- **power**: Power management, battery
- **emergency**: Wipe, kill switches (auto-quarantined)

### Safety Levels
- **SAFE**: Read-only, monitoring devices (90% confidence)
- **MONITORED**: Control and management devices (80% confidence)
- **CAUTION**: Write and modify operations (70% confidence)
- **QUARANTINED**: Dangerous operations (95-100% confidence)
- **UNKNOWN**: No classification match (50% confidence)

## Reports and Logs

### Discovery Report
**Location**: `/tmp/dsmil_ml_discovery_report.json`

**Contents:**
- Total devices discovered
- Safety level breakdown
- Interface breakdown (SMBIOS, ACPI, sysfs)
- Capability breakdown
- Optimal activation sequence
- Per-device details with ML confidence

### Workflow Report
**Location**: `/tmp/dsmil_integrated_workflow_report.json`

**Contents:**
- Workflow status and stage
- Devices discovered/activated/failed
- Thermal status
- Complete activation sequence
- Per-device activation results

### Logs
- `/tmp/dsmil_ml_discovery.log`: ML discovery operations
- `/tmp/dsmil_integrated_activation.log`: Integrated workflow operations
- `/tmp/dsmil_guided_activation.log`: Manual activation operations
- `/tmp/dsmil_operation_monitor.log`: Operation monitoring

## Integration with Existing Systems

### Compatible with:
- **dsmil_guided_activation.py**: Classic manual activation still available
- **dsmil_device_activation.py**: Uses same activation backend
- **dsmil_subsystem_controller.py**: Integrates with subsystem management
- **dsmil_operation_monitor.py**: Works alongside operation monitoring

### New Dependencies:
- **numpy**: For ML calculations (optional, graceful fallback)
- **smbios-bin**: For SMBIOS token scanning
- **acpidump**: For ACPI table scanning (optional)

## Troubleshooting

### Issue: No devices discovered
**Solution**:
- Ensure running as root (`sudo`)
- Install `libsmbios-bin`: `sudo apt install libsmbios-bin`
- Check BIOS settings for DSMIL support

### Issue: Activation fails
**Solution**:
- Verify DSMIL kernel driver is loaded: `lsmod | grep dsmil`
- Check device file exists: `ls -la /dev/dsmil*`
- Review logs: `tail -f /tmp/dsmil_integrated_activation.log`

### Issue: Thermal critical errors
**Solution**:
- Clean cooling system
- Reduce ambient temperature
- Activate fewer devices simultaneously
- Increase pause between activations

### Issue: ML confidence low
**Solution**:
- This is expected for unknown devices
- Use manual guided activation for low-confidence devices
- Review device signatures manually

## Performance

### Discovery Phase
- **SMBIOS scan**: 5-10 seconds
- **ACPI scan**: 1-2 seconds
- **sysfs scan**: < 1 second
- **ML classification**: < 1 second
- **Total**: ~10-15 seconds

### Activation Phase
- **Per-device activation**: 0.5-1 second
- **Thermal check**: < 0.1 second
- **84 devices**: ~1-2 minutes

### Monitoring Phase
- **Configurable**: Default 30 seconds
- **Check interval**: 5 seconds

## Future Enhancements

1. **Advanced ML Models**: Replace pattern matching with trained models
2. **Predictive Thermal Modeling**: Better thermal impact prediction
3. **Automated Optimization**: Self-tuning activation parameters
4. **Historical Analysis**: Learn from past activations
5. **Remote Monitoring**: Network-based monitoring and control
6. **GPU Acceleration**: Use NPU/GNA for ML inference

## Conclusion

The ML-Enhanced DSMIL Activation System provides a **mission-critical**, **intelligent**, and **safe** approach to hardware discovery and device activation. It combines automated discovery with machine learning classification to provide a smooth end-to-end workflow for activating DSMIL devices on the LAT5150 platform.

**Key Benefits:**
- ✅ **Automated**: No manual device enumeration needed
- ✅ **Intelligent**: ML predicts device capabilities and safety
- ✅ **Safe**: Multi-layer safety checks and thermal monitoring
- ✅ **Fast**: Complete workflow in ~2-3 minutes
- ✅ **Comprehensive**: Detailed reports and logs

## References

- `DSMIL_CURRENT_REFERENCE.md`: Complete DSMIL device database
- `DSMIL_DEVICE_CAPABILITIES.json`: Device capability definitions
- `PROJECT_OVERVIEW.md`: Overall project architecture
- `BUILD_ON_HARDWARE.md`: Hardware setup instructions

---

**Author**: LAT5150DRVMIL AI Platform
**Version**: 1.0.0
**Date**: 2025-11-12
**Classification**: DSMIL ML Integration
