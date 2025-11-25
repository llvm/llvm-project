# DSMIL Token Correlation Analysis Usage Guide

## Overview

The `analyze_token_correlation.py` script provides comprehensive correlation analysis between the 72 SMBIOS tokens (0x0480-0x04C7) and the 72 DSMIL devices discovered on the Dell Latitude 5450 MIL-SPEC system.

## Key Features

### 1. Architecture Analysis
- **6 Groups of 12 tokens** each matching DSMIL architecture
- **Sequential mapping** from token ID to Group/Device coordinates
- **Function hypothesis** based on Dell architectural patterns
- **Confidence scoring** for each token-to-function mapping

### 2. Safety Features
- **Thermal monitoring** before and during analysis
- **Dry-run mode** for safe analysis without token activation
- **Root privilege checking** for live analysis
- **System compatibility validation** (Dell hardware detection)

### 3. Output Formats
- **JSON report** for programmatic analysis
- **Human-readable report** for technical review
- **Console summary** for quick status

## Usage Examples

### Safe Analysis (Recommended First)
```bash
# Basic dry-run analysis
python3 analyze_token_correlation.py

# Verbose dry-run with custom output files
python3 analyze_token_correlation.py --verbose \
    --json my_correlation.json \
    --report my_analysis.txt
```

### Live Analysis (Requires Root)
```bash
# Live token accessibility testing
sudo python3 analyze_token_correlation.py --live --verbose

# Skip safety checks if needed
sudo python3 analyze_token_correlation.py --live --no-safety
```

## Output Interpretation

### Correlation Mapping
- **Group 0 (0x480-0x48B)**: Core system functions (highest confidence)
- **Group 1 (0x48C-0x497)**: Secondary system functions  
- **Group 2-5 (0x498-0x4C7)**: Extended/specialized functions (decreasing confidence)

### Function Categories
- **power_management**: Power state control, voltage regulation
- **thermal_control**: Temperature monitoring, fan control
- **security_module**: Hardware security, TPM integration
- **memory_control**: Memory configuration, timing
- **io_controller**: I/O port configuration
- **network_interface**: Network device control
- **storage_control**: Storage controller settings
- **display_control**: Display output configuration
- **audio_control**: Audio device settings
- **sensor_hub**: Environmental sensors
- **accelerometer**: Motion detection

### Confidence Levels
- **0.8-0.9**: High confidence (recommend testing first)
- **0.6-0.7**: Medium confidence (proceed with caution)  
- **0.3-0.5**: Low confidence (experimental only)
- **0.2**: Unknown function (research needed)

### Accessibility Status
- **✓**: Token accessible via SMBIOS interface
- **✗**: Token not accessible or protected

## Analysis Results Summary

From the dry-run analysis:

```
Total Tokens: 72 (0x480-0x4C7)
Average Confidence: 0.58
Accessible Tokens: 66.7% (48/72)
High Confidence: 6 tokens
```

### Priority Tokens for Testing

**Group 0 - Core System (Highest Priority)**:
- `0x480`: Power Management (90% confidence) - **Not Accessible**
- `0x481`: Thermal Control (90% confidence) - **Accessible** ⭐
- `0x482`: Security Module (80% confidence) - **Accessible** ⭐
- `0x483`: Memory Control (80% confidence) - **Not Accessible**

**Group 1 - Secondary System**:
- `0x48C`: Thermal Control (80% confidence) - **Not Accessible**
- `0x48D`: Power Management (80% confidence) - **Accessible** ⭐

## Safety Recommendations

### Before Live Testing
1. **Backup BIOS settings** completely
2. **Monitor system temperature** (keep below 85°C)
3. **Test individual tokens** never in batch
4. **Have emergency shutdown** procedure ready
5. **Document baseline values** before changes

### During Testing
1. **Monitor thermal status** continuously
2. **Test accessible high-confidence tokens first**
3. **Record all changes and effects**
4. **Be ready to revert changes immediately**
5. **Watch for system instability**

### Emergency Procedures
1. **Immediate thermal shutdown** if temp > 95°C
2. **BIOS reset** if system becomes unstable
3. **Power cycle** if system hangs
4. **CMOS clear** if BIOS corruption occurs

## Next Steps

1. **Start with Group 0 thermal control** (0x481) - accessible with 90% confidence
2. **Test security module** (0x482) in isolated environment
3. **Map accessible tokens** to actual DSMIL device behavior
4. **Correlate with thermal/power changes** during token modification
5. **Build comprehensive token → function database**

## Integration with DSMIL Research

This analysis provides the foundation for:
- **Precise DSMIL device control** via specific token manipulation
- **Hardware function mapping** for the 72-device architecture
- **Safe experimentation path** starting with highest-confidence tokens
- **Systematic device discovery** across all 6 groups

The correlation analysis enables targeted DSMIL device interaction while maintaining system safety through comprehensive monitoring and staged testing approaches.