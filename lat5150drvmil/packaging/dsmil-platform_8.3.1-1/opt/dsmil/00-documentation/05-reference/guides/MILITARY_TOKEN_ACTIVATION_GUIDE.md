# Military Token Activation Guide
**Dell Latitude 5450 MIL-SPEC - HARDWARE-DELL & SECURITY Agent Collaboration**

## Overview
Safe activation system for discovered military tokens with comprehensive safety protocols, thermal monitoring, and rollback capabilities.

## Files Created

### Main Activation Script
- **`activate_military_tokens.py`** - Primary activation script with full safety protocols
- **`validate_activation_safety.py`** - Pre-activation safety validation

## Target Tokens (Filtered for Safety)
```
0x8000 - Primary Command Interface
0x8014 - Secure Communications  
0x801E - Tactical Display Control
0x8028 - Power Management Unit
0x8032 - Memory Protection
0x803C - I/O Security Controller
0x8046 - Network Security Module
0x8050 - Storage Encryption
0x805A - Sensor Array
0x8064 - Auxiliary Systems
```

## Quarantined Tokens (NEVER ACTIVATED)
```
0x8009, 0x800A, 0x800B, 0x8019, 0x8029
```
These tokens are permanently quarantined due to instability or security risks.

## Safety Features

### Multi-Layer Protection
1. **Quarantine Filter** - Blocks dangerous tokens
2. **Thermal Monitoring** - Continuous temperature tracking
3. **System Checkpoints** - Complete rollback capability
4. **Activation Verification** - Confirms each step
5. **Stability Monitoring** - Post-activation system health

### Thermal Safety Limits
- **Warning Threshold**: 90°C
- **Critical Threshold**: 95°C  
- **Emergency Cutoff**: 100°C

## Usage Instructions

### Step 1: Safety Validation (REQUIRED)
```bash
# Run safety validation first
sudo python3 validate_activation_safety.py
```

**Exit codes:**
- `0` = SAFE_TO_PROCEED
- `1` = PROCEED_WITH_CAUTION  
- `2` = DO_NOT_PROCEED

### Step 2: Load DSMIL Module (if needed)
```bash
cd 01-source/kernel
sudo insmod dsmil-72dev.ko
```

### Step 3: Execute Activation Sequence
```bash
# Only proceed if safety validation passes
sudo python3 activate_military_tokens.py
```

## Activation Process

### Phase 1: Pre-activation Safety
- Verify DSMIL module loaded
- Check thermal conditions
- Create system checkpoint
- Backup all token states

### Phase 2: SMBIOS Token Activation  
- Filter quarantined tokens
- Activate each token with verification
- Monitor thermal impact
- Retry failed activations (up to 3 attempts)

### Phase 3: Dell WMI Security Features
- Enable SecureAdministrativeWorkstation
- Activate TpmSecurity, SecureBoot
- Configure ChasIntrusion detection
- Enable FirmwareTamperDet

### Phase 4: System Stability Monitoring
- 60-second thermal monitoring
- Stability verification
- Performance impact analysis

### Phase 5: Results & Verification
- Comprehensive activation report  
- Success rate analysis
- Rollback data preservation

## Expected Results

### Device Expansion Target
- **Current**: 29 devices controlled
- **Target**: 40+ devices controlled  
- **Expansion**: +11 military devices activated

### Success Criteria
- ✅ No quarantined tokens activated
- ✅ Thermal limits respected
- ✅ System remains stable
- ✅ Rollback capability preserved
- ✅ >50% token activation success

## Rollback Procedure

If activation causes issues:

```bash
# Automatic rollback using saved checkpoint
python3 rollback_activation.py rollback_data_TIMESTAMP.json
```

Or manual rollback:
```bash
# Disable individual tokens
sudo smbios-token-ctl --token-id=0x8000 --deactivate
```

## Output Files

### Generated During Execution
- `logs/activation_log_TIMESTAMP.txt` - Detailed execution log
- `logs/activation_results_TIMESTAMP.json` - Complete results data
- `checkpoints/activation_TIMESTAMP/` - System checkpoint
- `rollback_data_TIMESTAMP.json` - Rollback information
- `safety_assessment_TIMESTAMP.json` - Pre-activation validation

### Key Metrics Tracked
- Token activation success rate
- Thermal impact analysis  
- WMI security feature status
- System stability metrics
- Device expansion achieved

## Security Considerations

### Dell WMI Features Activated
- **SecureAdministrativeWorkstation** - Military workstation mode
- **TpmSecurity** - TPM hardware security
- **SecureBoot** - Verified boot process
- **ChasIntrusion** - Physical tampering detection
- **FirmwareTamperDet** - Firmware integrity monitoring
- **ThermalManagement** - Enhanced thermal control
- **PowerWarn** - Power anomaly detection

### Monitoring Integration
The activation integrates with existing monitoring systems:
- Thermal guardian integration
- DSMIL monitoring framework  
- Security audit logging
- Performance impact tracking

## Troubleshooting

### Common Issues

**"DSMIL module not loaded"**
```bash
cd 01-source/kernel
make clean && make
sudo insmod dsmil-72dev.ko
```

**"Dell WMI interface not found"**  
- Verify Dell system with WMI support
- Check BIOS settings for WMI enable

**"Thermal conditions unsafe"**
- Wait for system cooldown
- Check thermal guardian status
- Verify fan operation

**"Permission denied"**
- Run with sudo privileges
- Check /dev/dsmil* permissions

### Log Analysis
Check logs for detailed error information:
```bash
tail -f logs/activation_log_*.txt
cat logs/activation_results_*.json | jq '.failures'
```

## Verification Commands

### Check Activated Tokens
```bash
# Verify specific token activation
sudo smbios-token-ctl --token-id=0x8000 --get

# Check all target tokens
for token in 8000 8014 801E 8028 8032 803C 8046 8050 805A 8064; do
    echo "Token 0x$token:"
    sudo smbios-token-ctl --token-id=0x$token --get
done
```

### Monitor System Health
```bash
# Check thermal status
sensors

# Monitor DSMIL activity  
dmesg | grep -i dsmil | tail -10

# Check WMI security features
ls /sys/devices/virtual/firmware-attributes/dell-wmi-sysman/attributes/Secure*
```

## Mission Success Criteria

✅ **Primary Objective**: Expand device control from 29 to 40+ devices  
✅ **Safety Objective**: Zero quarantined token activations  
✅ **Thermal Objective**: Stay below 95°C throughout process  
✅ **Security Objective**: Enable military workstation features  
✅ **Stability Objective**: System remains stable post-activation  

---

**HARDWARE-DELL Agent**: Dell-specific optimization and token management  
**SECURITY Agent**: Safety protocols and risk mitigation  
**Created**: 2025-09-02 (Military Token Discovery Phase)