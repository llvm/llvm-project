# Dell MIL-SPEC Driver Kernel Parameters

## Overview

The Dell MIL-SPEC enhanced driver supports kernel parameters to enable and configure military-grade security subsystems based on Dell's actual DSMIL (Dell System Military) and MODE5 platform integrity enforcement systems. These parameters control 10 specialized DSMIL subsystems and 4 levels of MODE5 hardware lockdown.

⚠️ **CRITICAL WARNING**: MODE5 PARANOID and PARANOID_PLUS levels are **PERMANENT** and cannot be reversed!

## DSMIL Subsystem Parameters

### dsmil.enable
- **Type**: Boolean
- **Default**: `false` (disabled)
- **Usage**: `dsmil.enable=1`
- **Description**: Enables the DSMIL (Dell Security Military Implementation Layer) subsystem
- **Example**: `modprobe dell-milspec dsmil.enable=1`

### dsmil.mode
- **Type**: String
- **Default**: `"standard"`
- **Valid Values**: `standard`, `enhanced`, `paranoid`
- **Usage**: `dsmil.mode=enhanced`
- **Description**: Sets the DSMIL operation mode
  - `standard`: Basic military security features
  - `enhanced`: Advanced threat detection and response (RECOMMENDED)
  - `paranoid`: Maximum security with formal verification and continuous monitoring
- **Example**: `modprobe dell-milspec dsmil.enable=1 dsmil.mode=enhanced`

### dsmil.jrtc1
- **Type**: Boolean
- **Default**: `false` (disabled)
- **Usage**: `dsmil.jrtc1=1`
- **Description**: Enables JRTC1 (Joint Readiness Training Center) protocols for military joint operations
- **Requirements**: Requires `dsmil.enable=1`
- **Example**: `modprobe dell-milspec dsmil.enable=1 dsmil.jrtc1=1`

### dsmil.subsystems
- **Type**: Boolean Array (72 elements)
- **Default**: All `false` (disabled)  
- **Usage**: `dsmil.subsystems=1,1,0,1,0,0,0,0,0,0,0,0,...`
- **Description**: Enable specific DSMIL subsystems individually (72 total across 6 layers)
## DSMIL 6-Layer Architecture (72 Subsystems Total)

### Layer 0 (DSMIL0D0-DSMIL0DB) - Core Military Systems
- `DSMIL0D0`: Command & Control Interface
- `DSMIL0D1`: Tactical Communications  
- `DSMIL0D2`: Enhanced GPS/Navigation
- `DSMIL0D3`: Encrypted Storage Controller
- `DSMIL0D4`: Hardware Cryptography
- `DSMIL0D5`: Emergency Data Destruction
- `DSMIL0D6`: Tactical Sensor Integration
- `DSMIL0D7`: Power Management (Tactical)
- `DSMIL0D8`: Network Security Monitor
- `DSMIL0D9`: Mission Data Recorder
- `DSMIL0DA`: Advanced Threat Detection
- `DSMIL0DB`: Secure Boot Enhancement

### Layer 1 (DSMIL1D0-DSMIL1DB) - Enhanced Capabilities
- 12 additional specialized military subsystems

### Layer 2 (DSMIL2D0-DSMIL2DB) - Extended Operations
- 12 additional specialized military subsystems

### Layer 3 (DSMIL3D0-DSMIL3DB) - Advanced Features
- 12 additional specialized military subsystems

### Layer 4 (DSMIL4D0-DSMIL4DB) - Deep Integration
- 12 additional specialized military subsystems

### Layer 5 (DSMIL5D0-DSMIL5DB) - Maximum Security
- 12 additional specialized military subsystems

**Total**: 72 DSMIL subsystems across 6 hierarchical layers

### dsmil.all_subsystems
- **Type**: Boolean
- **Default**: `false` (disabled)
- **Usage**: `dsmil.all_subsystems=1`
- **Description**: Enable all 72 DSMIL subsystems across all 6 layers
- **⚠️ WARNING**: Activating all subsystems may impact system performance
- **Requirements**: Requires `dsmil.enable=1`
- **Example**: `modprobe dell-milspec dsmil.enable=1 dsmil.all_subsystems=1`

### dsmil.layer
- **Type**: Integer (0-5)
- **Default**: `0` (Layer 0 only)
- **Usage**: `dsmil.layer=2`
- **Description**: Enable all subsystems up to specified layer
- **Layer Progression**:
  - `0`: Core military systems (12 subsystems)
  - `1`: + Enhanced capabilities (24 total)
  - `2`: + Extended operations (36 total)
  - `3`: + Advanced features (48 total)
  - `4`: + Deep integration (60 total)
  - `5`: + Maximum security (72 total)
- **Example**: `modprobe dell-milspec dsmil.enable=1 dsmil.layer=3`

## MODE5 Subsystem Parameters

### mode5.enable
- **Type**: Boolean
- **Default**: `false` (disabled)
- **Usage**: `mode5.enable=1`
- **Description**: Enables the MODE5 cryptographic subsystem for secure communications
- **Example**: `modprobe dell-milspec mode5.enable=1`

### mode5.level
- **Type**: String
- **Default**: `"standard"`
- **Valid Values**: `standard`, `enhanced`, `paranoid`, `paranoid_plus`
- **Usage**: `mode5.level=enhanced`
- **Description**: Sets the MODE5 platform integrity enforcement level
  - `standard`: VM migration allowed, reversible configuration
  - `enhanced`: VMs locked to hardware signature, semi-permanent (requires Dell service)
  - `paranoid`: **PERMANENT** hardware lockdown, no VM migration, restricted access
  - `paranoid_plus`: **PERMANENT** + automatic secure wipe on intrusion detection
- **⚠️ WARNING**: `paranoid` and `paranoid_plus` modes are **IRREVERSIBLE**
- **🚨 CRITICAL**: `paranoid_plus` will **DESTROY DATA** on chassis opening or tampering
- **Example**: `modprobe dell-milspec mode5.enable=1 mode5.level=enhanced`

### mode5.migration
- **Type**: Boolean
- **Default**: `false` (disabled)
- **Usage**: `mode5.migration=1`
- **Description**: Controls VM migration capability (only effective in `standard` mode)
- **Requirements**: Requires `mode5.enable=1`
- **Note**: Ignored in `enhanced`, `paranoid`, and `paranoid_plus` modes
- **Example**: `modprobe dell-milspec mode5.enable=1 mode5.migration=0`

## DSMIL Subsystem Details

### Command & Control (DSMIL0D0)
Primary military system coordinator managing communication between all subsystems, authentication/authorization, and overall security state control.

### Tactical Communications (DSMIL0D1) 
Encrypted communication channels with military radio integration, satellite communication support, and mesh networking capabilities for field operations.

### Enhanced GPS/Navigation (DSMIL0D2)
Military-grade GPS with SAASM/M-Code support, inertial navigation backup, terrain mapping integration, and anti-spoofing/jamming protection.

### Encrypted Storage Controller (DSMIL0D3)
Hardware-level encryption beyond standard implementations, multi-level security compartments, rapid secure erase capabilities, and classified data handling.

### Hardware Cryptography (DSMIL0D4)
Military encryption algorithms, comprehensive key management system, certificate handling, and secure boot extensions.

### Emergency Data Destruction (DSMIL0D5)
Multiple wipe algorithms (DoD 5220.22-M compliance), thermite-equivalent data destruction, remote trigger capability, and tamper-activated emergency wipe.

### Tactical Sensor Integration (DSMIL0D6)
Environmental sensor inputs, biometric authentication systems, CBRN (Chemical, Biological, Radiological, Nuclear) detection interfaces, and motion/vibration monitoring.

### Power Management - Tactical (DSMIL0D7)
Extended battery operations, power signature masking, emergency power modes, and solar/alternative charging support for field deployment.

### Network Security Monitor (DSMIL0D8)
Advanced intrusion detection/prevention, traffic analysis resistance, covert channel detection, and network isolation controls.

### Mission Data Recorder (DSMIL0D9)
Black box functionality with forensic data capture, chain of custody maintenance, and encrypted audit trail recording.

## Legacy Parameters

### milspec_debug
- **Type**: Unsigned Integer (bitmask)
- **Default**: `0`
- **Usage**: `milspec_debug=0x7`
- **Description**: Debug level bitmask for troubleshooting
- **Values**:
  - `0x1`: Basic debug messages
  - `0x2`: Hardware access debug
  - `0x4`: Crypto operations debug
  - `0x8`: Event logging debug

### milspec_force
- **Type**: Boolean
- **Default**: `false`
- **Usage**: `milspec_force=1`
- **Description**: Force load driver on non-Dell systems (testing only)

## Complete Usage Examples

### Basic DSMIL Activation
```bash
# Enable DSMIL with enhanced mode and JRTC1 protocols
modprobe dell-milspec dsmil.enable=1 dsmil.mode=enhanced dsmil.jrtc1=1
```

### Complete Military Configuration
```bash
# Enable both DSMIL and MODE5 with maximum security
modprobe dell-milspec \
    dsmil.enable=1 \
    dsmil.mode=enhanced \
    dsmil.jrtc1=1 \
    mode5.enable=1 \
    mode5.level=standard \
    mode5.migration=1
```

### Boot-time Configuration
Add to kernel command line or `/etc/modprobe.d/dell-milspec.conf`:
```
options dell-milspec dsmil.enable=1 dsmil.mode=enhanced mode5.enable=1 mode5.level=standard
```

## Kernel Command Line Usage

For early boot activation, add parameters to kernel command line:
```
linux ... dsmil.enable=1 dsmil.mode=enhanced dsmil.jrtc1=1 mode5.enable=1 mode5.level=standard mode5.migration=1
```

## Security Considerations

- **Production Systems**: Always use `dsmil.mode=enhanced` or `dsmil.mode=paranoid`
- **Key Migration**: Enable `mode5.migration=1` for systems requiring key rotation
- **JRTC1 Protocols**: Only enable `dsmil.jrtc1=1` in joint military environments
- **Debug Mode**: Never use `milspec_debug` in production deployments
- **Paranoid Mode**: Use `dsmil.mode=paranoid` for maximum security in high-threat environments

## Verification

After loading the module, verify activation in kernel logs:
```bash
dmesg | grep "MIL-SPEC"
```

Expected output:
```
[    X.XXXXXX] MIL-SPEC: DSMIL subsystem enabled (mode: enhanced)
[    X.XXXXXX] MIL-SPEC: JRTC1 joint readiness protocols enabled
[    X.XXXXXX] MIL-SPEC: MODE5 subsystem enabled (level: standard, migration: enabled)
[    X.XXXXXX] MIL-SPEC: Military subsystem status - DSMIL: ACTIVE, MODE5: ACTIVE
```

## Status Monitoring

Check subsystem status via sysfs:
```bash
cat /sys/class/milspec/milspec/dsmil_status
cat /sys/class/milspec/milspec/mode5_status
```

## Troubleshooting

1. **Module Load Failures**: Check hardware compatibility with `lspci` and `dmidecode`
2. **Parameter Validation**: Invalid parameters fall back to safe defaults
3. **Debug Output**: Use `milspec_debug=0x7` for detailed troubleshooting
4. **Force Loading**: Use `milspec_force=1` for testing on non-Dell hardware

## Related Documentation

- [Dell MIL-SPEC Driver Architecture](../02-analysis/architecture/)
- [Security Implementation Plan](../01-planning/phase-1-core/ADVANCED-SECURITY-PLAN.md)
- [Hardware Integration Guide](../01-planning/phase-1-core/KERNEL-INTEGRATION-PLAN.md)