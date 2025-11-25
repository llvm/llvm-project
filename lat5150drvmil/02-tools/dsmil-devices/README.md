# DSMIL Device Integration Framework

**Unified Device Integration**

Unified Python framework for integrating DSMIL devices on the Dell Latitude 5450 MIL-SPEC platform.

## Overview

This integration framework provides high-level Python interfaces to 22 critical DSMIL hardware devices, enabling secure system management, cryptographic operations, security monitoring, and tactical operations without requiring direct hardware access.

### Integrated Devices

| Device ID | Name | Group | Risk Level | Description |
|-----------|------|-------|------------|-------------|
| **0x8000** | TPM Control | Core Security | MONITORED | TPM 2.0 cryptographic operations |
| **0x8001** | Boot Security | Core Security | MONITORED | Secure boot and boot chain integrity |
| **0x8002** | Credential Vault | Core Security | MONITORED | Secure credential storage |
| **0x8003** | Audit Log | Core Security | SAFE | Security audit logging & compliance tracking |
| **0x8004** | Event Logger | Core Security | SAFE | System event logging & monitoring |
| **0x8005** | Performance Monitor | Core Security | SAFE | Performance monitoring & TPM/HSM interface |
| **0x8006** | Thermal Sensor | Core Security | SAFE | System thermal monitoring & management |
| **0x8007** | Power State | Core Security | MONITORED | Power management & ACPI state control |
| **0x8008** | Emergency Response | Core Security | MONITORED | Secure boot validation & emergency response |
| **0x8010** | Intrusion Detection | Extended Security | SAFE | Physical intrusion monitoring |
| **0x8013** | Key Management | Extended Security | MONITORED | Cryptographic key lifecycle management |
| **0x8014** | Certificate Store | Extended Security | SAFE | PKI certificate management |
| **0x8016** | VPN Controller | Extended Security | MONITORED | VPN tunnel management |
| **0x8017** | Remote Access | Extended Security | MONITORED | Remote access control & session management |
| **0x8018** | Pre-Isolation | Extended Security | MONITORED | Network pre-isolation & threat containment |
| **0x801A** | Port Security | Extended Security | MONITORED | Physical port security control |
| **0x801B** | Wireless Security | Extended Security | MONITORED | Wireless communication security |
| **0x801E** | Tactical Display | Network/Comms | MONITORED | Military-grade display security |
| **0x802A** | Network Monitor | Network/Comms | SAFE | Network traffic monitoring & analysis |
| **0x802B** | Packet Filter | Network/Comms | MONITORED | Hardware packet filtering |
| **0x8050** | Storage Encryption | Storage/Data | MONITORED | Full disk encryption & SED management |
| **0x805A** | Sensor Array | Peripheral/Sensors | SAFE | Multi-sensor monitoring & fusion |

**Total: 22 devices (20.4% of 108 total DSMIL devices)**

## Features

- ✅ **Interactive Menu System** - User-friendly TUI for device management
- ✅ **Unified Device API** - Consistent interface across all devices
- ✅ **Type-Safe Device Registry** - Centralized device management
- ✅ **Risk-Based Access Control** - Enforces device risk policies
- ✅ **Hardware Abstraction** - No direct hardware dependency
- ✅ **Simulated Operation** - Full functionality without hardware
- ✅ **Comprehensive Status Reporting** - Detailed device state information
- ✅ **Structured Logging** - Operation tracking and audit trails
- ✅ **Example Usage** - Complete API documentation

## Architecture

```
dsmil-devices/
├── lib/                              # Core libraries
│   ├── device_base.py                # Base device class
│   └── device_registry.py            # Device registry
├── devices/                          # Device implementations (22 devices)
│   ├── device_0x8000_tpm_control.py
│   ├── device_0x8001_boot_security.py
│   ├── device_0x8002_credential_vault.py
│   ├── device_0x8003_audit_log.py
│   ├── device_0x8004_event_logger.py
│   ├── device_0x8005_performance_monitor.py
│   ├── device_0x8006_thermal_sensor.py
│   ├── device_0x8007_power_state.py
│   ├── device_0x8008_emergency_response.py
│   ├── device_0x8010_intrusion_detection.py
│   ├── device_0x8013_key_management.py
│   ├── device_0x8014_certificate_store.py
│   ├── device_0x8016_vpn_controller.py
│   ├── device_0x8017_remote_access.py
│   ├── device_0x8018_pre_isolation.py
│   ├── device_0x801A_port_security.py
│   ├── device_0x801B_wireless_security.py
│   ├── device_0x801E_tactical_display.py
│   ├── device_0x802A_network_monitor.py
│   ├── device_0x802B_packet_filter.py
│   ├── device_0x8050_storage_encryption.py
│   └── device_0x805A_sensor_array.py
├── examples/                         # Usage examples
│   ├── basic_usage.py
│   └── new_devices_usage.py
├── docs/                             # Documentation
├── dsmil_integration.py              # Main integration module
├── dsmil_menu.py                     # Interactive TUI menu
├── dsmil_discover.py                 # Local system discovery
├── dsmil_probe.py                    # Device functional prober
└── README.md                         # This file
```

## Quick Start

### Installation

No installation required - pure Python 3.6+ framework.

```bash
cd /home/user/LAT5150DRVMIL/02-tools/dsmil-devices
```

### Basic Usage

```python
#!/usr/bin/env python3
from dsmil_integration import get_device, initialize_all_devices

# Initialize all devices
results = initialize_all_devices()

# Get TPM Control device
tpm = get_device(0x8000)
if tpm:
    # Initialize
    result = tpm.initialize()
    print(f"TPM initialized: {result.success}")

    # Get status
    status = tpm.get_status()
    print(f"TPM version: {status['version']}")

    # Read register
    result = tpm.read_register("CAPABILITIES")
    print(f"Capabilities: {result.data}")

    # Generate random bytes
    result = tpm.get_random(32)
    print(f"Random data: {result.data['data']}")
```

### Command-Line Interface

```bash
# Show integration summary
python3 dsmil_integration.py --summary

# List all devices
python3 dsmil_integration.py --list

# Initialize all devices
python3 dsmil_integration.py --initialize

# Get device info
python3 dsmil_integration.py --device 0x8000

# Test a device
python3 dsmil_integration.py --test 0x8001
```

### Interactive Menu System

The interactive menu provides a user-friendly TUI (Text User Interface) for managing devices:

```bash
# Launch interactive menu
python3 dsmil_menu.py
```

**Menu Features:**
- **Browse All Devices** - View all 22 devices grouped by category
- **Browse by Group** - Navigate devices by functional group (Core Security, Extended Security, etc.)
- **Browse New Devices** - Quick access to recently integrated devices
- **System Status** - View integration summary and device statistics
- **Initialize All** - Initialize all devices in one operation
- **Quick Device Access** - Jump to a device by hex ID (e.g., 8003, 8017, 802A)

**Device Operations:**
Each device has a dedicated submenu with device-specific operations:
- Credential Vault (0x8002): List credentials, vault policy, capacity info, access log
- Audit Log (0x8003): Recent entries, filter by severity/category, audit summary
- Event Logger (0x8004): Recent events, filter by level, error events, summary
- Performance Monitor (0x8005): Get metrics, thermal status, HSM status, crypto performance
- Thermal Sensor (0x8006): All temperatures, CPU temperature, thermal summary, alerts
- Power State (0x8007): Power summary, battery info, performance info, state transitions
- Emergency Response (0x8008): Validate boot chain, integrity report, emergency status, alerts
- Key Management (0x8013): List keys, storage summary, key summary, rotation history
- VPN Controller (0x8016): List tunnels, encryption info, statistics
- Remote Access (0x8017): List sessions, access history, failed attempts, security config
- Pre-Isolation (0x8018): Isolation status, isolated systems, threat assessment, network zones
- Tactical Display (0x801E): List displays, security config, TEMPEST status, protection features
- Network Monitor (0x802A): List interfaces, traffic stats, protocol breakdown, bandwidth usage, anomalies
- Packet Filter (0x802B): Get filter rules, statistics
- Storage Encryption (0x8050): List volumes/SEDs, encryption config, OPAL support
- Sensor Array (0x805A): List sensors, environmental summary, security status, fusion data

**Navigation:**
- Use number keys to select options
- Press `[0]` to go back to previous menu
- All operations display results in formatted output
- Press `Enter` to continue after viewing results

### Local System Discovery

Discover what DSMIL hardware and drivers are actually present on the local system:

```bash
# Run system discovery
python3 dsmil_discover.py

# Run with sudo for full access
sudo python3 dsmil_discover.py
```

**Discovery Checks:**
- **Kernel Modules** - Scans loaded modules (lsmod) for DSMIL/Dell drivers
- **Device Nodes** - Checks /dev/ for DSMIL device nodes
- **Kernel Messages** - Analyzes dmesg for DSMIL references
- **PCI Devices** - Lists relevant PCI devices (lspci)
- **ACPI Tables** - Enumerates ACPI tables in /sys/firmware
- **SMI Interface** - Checks SMI port accessibility (0xB2/0xB3)
- **Running Processes** - Finds DSMIL-related processes
- **Firmware Files** - Locates Dell/DSMIL firmware in /lib/firmware
- **System Devices** - Scans /sys for device information
- **Proc Devices** - Checks /proc for device entries
- **Framework Status** - Shows integration framework status

**Discovered Information:**
- System DMI information (vendor, product, BIOS version)
- Loaded vs available kernel modules
- Device node major/minor numbers and permissions
- SMI interface availability and restrictions
- Framework integration status vs actual hardware
- Recent kernel messages related to DSMIL

**Use Cases:**
- Verify DSMIL driver installation
- Check hardware presence before integration
- Troubleshoot device access issues
- Compare framework vs actual hardware
- Security audit of device accessibility

### Device Functional Probing

Test device functionality and discover what each device does by probing operations:

```bash
# Run functional prober
python3 dsmil_probe.py
```

**Probing Features:**
- **Device Purpose** - Shows what each device does in detail
- **Safe Operation Testing** - Tests read-only and safe operations only
- **Quarantine Protection** - Automatically skips 5 dangerous quarantined devices
- **Capability Detection** - Lists device capabilities (read/write, DMA, etc.)
- **Register Map Display** - Shows device register layout and descriptions
- **Operation Results** - Tests each device-specific operation
- **Success Rate** - Calculates operational success percentage
- **Functional Status** - Categorizes devices as FUNCTIONAL, PARTIAL, or NON-FUNCTIONAL

**Quarantined Devices (Auto-Skipped):**
```
⛔ 0x8009 - Self-Destruct (Destructive)
⛔ 0x800A - Secure Erase (Destructive)
⛔ 0x800B - Emergency Lockdown (Destructive)
⛔ 0x8019 - Remote Disable (Destructive)
⛔ 0x8029 - System Reset (Destructive)
```

**Tested Operations by Device:**
- **0x8005 (Performance Monitor)**: 8 operations including metrics, thermal, HSM, crypto performance
- **0x8016 (VPN Controller)**: 5 operations including tunnel listing, encryption info
- **0x801E (Tactical Display)**: 9 operations including displays, security, TEMPEST, protection
- **0x8050 (Storage Encryption)**: 9 operations including volumes, SEDs, encryption config, OPAL
- **0x805A (Sensor Array)**: 9 operations including sensors, environmental, security, fusion

**Report Sections:**
1. **Quick Summary Table** - Device ID, name, status, success rate at a glance
2. **Quarantined Devices** - Lists permanently blocked devices with reasons
3. **Device Details** - Comprehensive per-device reports with:
   - Purpose and description
   - Initialization status
   - Capabilities list
   - Register map (first 5 registers)
   - Operation test results
   - Working vs failed operations
   - Overall functional status
4. **Statistics** - Total operations, success rates, device categories

**Use Cases:**
- Understand what each device module does
- Test device functionality without hardware
- Verify device operations before deployment
- Identify partially functional devices
- Debug device integration issues
- Generate device capability documentation

## Device Guides

### Device 0x8000: TPM Control

**Purpose:** TPM 2.0 cryptographic operations, secure key storage, and attestation.

**Key Operations:**
- Key generation (RSA, ECC, post-quantum)
- PCR extend/read
- Data sealing/unsealing
- Hardware random number generation

**Example:**
```python
from dsmil_integration import get_device

tpm = get_device(0x8000)
tpm.initialize()

# Read PCR
result = tpm.read_pcr(0)
print(f"PCR[0]: {result.data['hex']}")

# Generate key
result = tpm.generate_key(algorithm=tpm.TPM2Algorithm.RSA_2048)
print(f"Key handle: {result.data['key_handle']}")

# Get random bytes
result = tpm.get_random(32)
print(f"Random: {result.data['data']}")
```

**Registers:**
- `STATUS` - TPM status (RO)
- `CAPABILITIES` - TPM capabilities bitmap (RO)
- `VERSION` - TPM version (RO)
- `ERROR_CODE` - Last error code (RO)

### Device 0x8001: Boot Security

**Purpose:** Secure boot validation and boot chain integrity monitoring.

**Key Operations:**
- Get boot policy
- Get boot measurements
- Verify boot stages
- Check signature status

**Example:**
```python
from dsmil_integration import get_device

boot_sec = get_device(0x8001)
boot_sec.initialize()

# Get boot policy
result = boot_sec.get_boot_policy()
print(f"Boot policy: {result.data['flags']}")

# Get boot measurements
result = boot_sec.get_boot_measurements()
print(f"All measurements valid: {result.data['all_valid']}")

# Get boot chain summary
result = boot_sec.get_boot_chain_summary()
print(f"Secure boot: {result.data['secure_boot']}")
```

**Registers:**
- `BOOT_STATUS` - Boot security status (RO)
- `BOOT_POLICY` - Active boot policy (RO)
- `BOOT_STAGE` - Current boot stage (RO)
- `ROLLBACK_INDEX` - Rollback protection index (RO)

### Device 0x8002: Credential Vault

**Purpose:** Secure storage of credentials, keys, passwords, and sensitive data.

**Key Operations:**
- List credentials
- Get credential info
- Retrieve credentials (requires unlock)
- Lock/unlock vault

**Example:**
```python
from dsmil_integration import get_device

vault = get_device(0x8002)
vault.initialize()

# Unlock vault (simulated authentication)
result = vault.unlock_vault(auth_token="simulated_token")
if result.success:
    # List credentials
    result = vault.list_credentials()
    print(f"Stored credentials: {len(result.data['credentials'])}")

    # Get capacity info
    result = vault.get_capacity_info()
    print(f"Usage: {result.data['usage_percent']}%")

    # Lock vault
    vault.lock_vault()
```

**Registers:**
- `VAULT_STATUS` - Vault status (RO)
- `VAULT_POLICY` - Security policy (RO)
- `CREDENTIAL_COUNT` - Stored credentials (RO)
- `LOCK_STATUS` - Lock status (RO)

### Device 0x8010: Intrusion Detection

**Purpose:** Physical intrusion detection and security monitoring.

**Key Operations:**
- Get sensor states
- Get intrusion events
- Monitor chassis security

**Example:**
```python
from dsmil_integration import get_device

ids = get_device(0x8010)
ids.initialize()

# Get sensor states
result = ids.get_sensor_states()
print(f"All sensors secure: {result.data['all_secure']}")

# Get intrusion events
result = ids.get_intrusion_events(limit=10)
print(f"Recent events: {len(result.data['events'])}")
```

**Registers:**
- `IDS_STATUS` - IDS status (RO)
- `EVENT_COUNT` - Total events (RO)
- `SENSOR_STATUS` - Sensor bitmap (RO)
- `CHASSIS_STATUS` - Chassis status (RO)

### Device 0x8014: Certificate Store

**Purpose:** PKI certificate storage and management.

**Key Operations:**
- List certificates
- Get certificate details
- Verify certificate chains

**Example:**
```python
from dsmil_integration import get_device

cert_store = get_device(0x8014)
cert_store.initialize()

# List certificates
result = cert_store.list_certificates()
for cert in result.data['certificates']:
    print(f"{cert['subject']}: {cert['valid']}")
```

**Registers:**
- `STORE_STATUS` - Store status (RO)
- `CERT_COUNT` - Certificate count (RO)
- `CAPACITY` - Max certificates (RO)
- `CRL_COUNT` - CRL count (RO)

### Device 0x801A: Port Security

**Purpose:** Physical port security and access control.

**Key Operations:**
- Get port list
- Monitor port access
- Enforce port policies

**Example:**
```python
from dsmil_integration import get_device

port_sec = get_device(0x801A)
port_sec.initialize()

# Get port list
result = port_sec.get_port_list()
for port in result.data['ports']:
    print(f"{port['name']}: {'enabled' if port['enabled'] else 'disabled'}")
```

**Registers:**
- `PORT_STATUS` - Port status (RO)
- `POLICY` - Security policy (RO)
- `ACTIVE_PORTS` - Active ports (RO)
- `BLOCKED_COUNT` - Blocked attempts (RO)

### Device 0x801B: Wireless Security

**Purpose:** Wireless communication security control.

**Key Operations:**
- Get interface list
- Monitor wireless status
- Enforce TEMPEST compliance

**Example:**
```python
from dsmil_integration import get_device

wireless = get_device(0x801B)
wireless.initialize()

# Get interface list
result = wireless.get_interface_list()
for iface in result.data['interfaces']:
    print(f"{iface['name']}: {iface['enabled']}")

# Check status
status = wireless.get_status()
print(f"TEMPEST compliant: {status['tempest_compliant']}")
```

**Registers:**
- `WIRELESS_STATUS` - Wireless status (RO)
- `ENABLED_INTERFACES` - Enabled count (RO)
- `ENCRYPTION_STATUS` - Encryption status (RO)
- `RF_EMISSIONS` - RF emission level (RO)

### Device 0x802B: Packet Filter

**Purpose:** Hardware-accelerated packet filtering and inspection.

**Key Operations:**
- Get filter rules
- Monitor filter statistics
- Configure DPI/IPS

**Example:**
```python
from dsmil_integration import get_device

pkt_filter = get_device(0x802B)
pkt_filter.initialize()

# Get filter rules
result = pkt_filter.get_filter_rules()
for rule in result.data['rules']:
    print(f"Rule {rule['rule_id']}: {rule['protocol']}:{rule['port']}")

# Get statistics
stats = pkt_filter.get_statistics()
print(f"Packets filtered: {stats['packets_filtered']}")
```

**Registers:**
- `FILTER_STATUS` - Filter status (RO)
- `RULE_COUNT` - Active rules (RO)
- `PACKETS_FILTERED` - Total filtered (RO)
- `PACKETS_BLOCKED` - Total blocked (RO)

### Device 0x8005: Performance Monitor

**Purpose:** System performance monitoring and TPM/HSM interface coordination.

**Key Operations:**
- Get current performance metrics
- Monitor thermal status
- Track cryptographic operations
- Check HSM/TPM activity

**Example:**
```python
from dsmil_integration import get_device

perf_mon = get_device(0x8005)
perf_mon.initialize()

# Get current metrics
result = perf_mon.get_current_metrics()
print(f"CPU: {result.data['metrics']['cpu_usage']:.1f}%")
print(f"Crypto ops/s: {result.data['metrics']['crypto_ops']}")

# Get thermal status
result = perf_mon.get_thermal_status()
print(f"Temperature: {result.data['temperature_celsius']:.1f}°C")
```

**Registers:**
- `MONITOR_STATUS` - Monitor status (RO)
- `CPU_USAGE` - CPU usage (RO)
- `MEMORY_USAGE` - Memory usage (RO)
- `CRYPTO_OPS` - Crypto operations/sec (RO)
- `TPM_ACTIVITY` - TPM operations/sec (RO)

### Device 0x8016: VPN Controller

**Purpose:** VPN tunnel management and secure routing for encrypted communications.

**Key Operations:**
- List and manage VPN tunnels
- Configure encryption settings
- Monitor tunnel status
- FIPS mode compliance

**Example:**
```python
from dsmil_integration import get_device

vpn = get_device(0x8016)
vpn.initialize()

# List tunnels
result = vpn.list_tunnels()
for tunnel in result.data['tunnels']:
    print(f"{tunnel['name']}: {tunnel['status']}")

# Get encryption info
result = vpn.get_encryption_info()
print(f"FIPS mode: {result.data['fips_mode']}")
```

**Registers:**
- `VPN_STATUS` - VPN status (RO)
- `ACTIVE_TUNNELS` - Active tunnel count (RO)
- `ENCRYPTION_STATUS` - Encryption status (RO)
- `THROUGHPUT` - Total throughput (RO)

### Device 0x801E: Tactical Display

**Purpose:** Military-grade display security with content protection and TEMPEST compliance.

**Key Operations:**
- Manage display security zones
- Configure protection levels
- Monitor TEMPEST compliance
- Control tactical overlays

**Example:**
```python
from dsmil_integration import get_device

display = get_device(0x801E)
display.initialize()

# Get security config
result = display.get_security_config()
print(f"Security zone: {result.data['security_zone']}")
print(f"Capture blocking: {result.data['capture_blocking']}")

# Get TEMPEST status
result = display.get_tempest_status()
print(f"Compliant: {result.data['compliant']}")
```

**Registers:**
- `DISPLAY_STATUS` - Display status (RO)
- `SECURITY_ZONE` - Security zone level (RW)
- `PROTECTION_LEVEL` - Content protection (RW)
- `TEMPEST_STATUS` - TEMPEST compliance (RO)

### Device 0x8050: Storage Encryption

**Purpose:** Full disk encryption and self-encrypting drive management.

**Key Operations:**
- Manage encrypted volumes
- Control SED drives
- Configure encryption algorithms
- Monitor encryption status

**Example:**
```python
from dsmil_integration import get_device

storage = get_device(0x8050)
storage.initialize()

# List volumes
result = storage.list_volumes()
print(f"Encrypted: {result.data['encrypted']}/{result.data['total']}")

# List SED drives
result = storage.list_sed_drives()
for drive in result.data['drives']:
    print(f"{drive['model']}: {drive['status']}")
```

**Registers:**
- `ENCRYPTION_STATUS` - Encryption status (RO)
- `ENCRYPTED_VOLUMES` - Encrypted count (RO)
- `SED_STATUS` - SED drive status (RO)
- `ALGORITHM` - Active algorithm (RW)

### Device 0x805A: Sensor Array

**Purpose:** Multi-sensor monitoring with fusion for situational awareness.

**Key Operations:**
- Monitor environmental sensors
- Detect security threats
- Sensor data fusion
- Radiation monitoring

**Example:**
```python
from dsmil_integration import get_device

sensors = get_device(0x805A)
sensors.initialize()

# Get environmental summary
result = sensors.get_environmental_summary()
print(f"Temp: {result.data['temperature']['celsius']:.1f}°C")
print(f"Humidity: {result.data['humidity']['percent']:.1f}%")

# Get security summary
result = sensors.get_security_summary()
print(f"Status: {result.data['overall_status']}")

# Get sensor fusion
result = sensors.get_fusion_data()
print(f"Threat level: {result.data['situational_awareness']['threat_level']}")
```

**Registers:**
- `ARRAY_STATUS` - Sensor array status (RO)
- `ACTIVE_SENSORS` - Active sensor count (RO)
- `TEMPERATURE` - Temperature (RO)
- `RADIATION_LEVEL` - Radiation level (RO)
- `TAMPER_STATUS` - Tamper detection (RO)

## API Reference

### Device Base Class

All devices inherit from `DSMILDeviceBase`:

```python
class DSMILDeviceBase:
    def initialize() -> OperationResult
    def get_capabilities() -> List[DeviceCapability]
    def get_status() -> Dict[str, Any]
    def read_register(register: str) -> OperationResult
    def write_register(register: str, value: int) -> OperationResult
    def get_register_map() -> Dict[str, Dict]
    def get_statistics() -> Dict[str, Any]
```

### Operation Result

All operations return an `OperationResult` object:

```python
class OperationResult:
    success: bool       # Operation succeeded
    data: Any          # Result data
    error: str         # Error message (if failed)
    duration: float    # Operation duration in seconds
    timestamp: float   # Operation timestamp
```

### Device Registry

Access devices through the global registry:

```python
from dsmil_integration import (
    get_device,              # Get device by ID
    get_all_devices,         # Get all devices
    get_devices_by_group,    # Get devices in group
    get_devices_by_risk,     # Get devices by risk level
    list_devices,            # List device info
    get_integration_summary  # Get summary
)
```

## Safety and Security

### Risk Levels

- **SAFE**: Read operations are completely safe
- **MONITORED**: Read operations safe, write operations logged
- **CAUTION**: Some operations may affect system state
- **RISKY**: Operations require careful consideration
- **QUARANTINED**: Device permanently blocked

### Access Control

The framework enforces access control based on device risk levels:

```python
# Safe devices - unrestricted access
device = get_device(0x8010)  # Intrusion Detection - SAFE

# Monitored devices - access logged
device = get_device(0x8000)  # TPM Control - MONITORED

# Quarantined devices - access denied
device = get_device(0x8009)  # Returns None - QUARANTINED
```

### Best Practices

1. **Always initialize devices** before use
2. **Check operation results** for success/failure
3. **Handle errors gracefully** with try/except
4. **Lock sensitive devices** (vault) after use
5. **Monitor device statistics** for unusual activity
6. **Use read-only operations** when possible

## Testing

### Unit Testing

Each device includes comprehensive testing:

```bash
python3 dsmil_integration.py --test 0x8000  # Test TPM Control
python3 dsmil_integration.py --test 0x8001  # Test Boot Security
python3 dsmil_integration.py --test 0x8002  # Test Credential Vault
```

### Integration Testing

Test all devices together:

```bash
python3 dsmil_integration.py --initialize
```

## Troubleshooting

### Device Not Found

```python
device = get_device(0x8000)
if device is None:
    print("Device not registered or disabled")
```

### Initialization Failed

```python
result = device.initialize()
if not result.success:
    print(f"Initialization failed: {result.error}")
```

### Operation Failed

```python
result = device.read_register("STATUS")
if not result.success:
    print(f"Read failed: {result.error}")
```

## Development

### Adding New Devices

1. Create device module in `devices/`
2. Inherit from `DSMILDeviceBase`
3. Implement required methods
4. Register in `dsmil_integration.py`

### Extending Functionality

Add device-specific operations as methods:

```python
class MyDevice(DSMILDeviceBase):
    def custom_operation(self, param) -> OperationResult:
        # Implementation
        return OperationResult(True, data=result)
```

## Performance

All operations are designed for minimal overhead:

- Device initialization: < 1ms
- Register reads: < 0.1ms
- Status queries: < 0.1ms
- Complex operations: < 10ms

## Classification

**UNCLASSIFIED // FOR OFFICIAL USE ONLY**

This integration framework is designed for military-grade hardware. Follow security protocols and access control policies.

## Version History

- **v1.4.1** (2025-01-06): Enhanced menu operations
  - Added Credential Vault (0x8002) operations to interactive menu
  - Added Packet Filter (0x802B) operations to interactive menu
  - Performance Monitor (0x8005) already included
  - Total: 7 devices with dedicated operation menus

- **v1.4.0** (2025-01-06): Device functional probing
  - Added device functional prober (`dsmil_probe.py`)
  - Tests safe operations on all devices
  - Displays device purpose and capabilities
  - Automatic quarantine protection (skips 5 dangerous devices)
  - Register map display
  - Operation success rate calculation
  - Comprehensive functional reports
  - Quick summary tables

- **v1.3.0** (2025-01-06): Local system discovery
  - Added local system discovery script (`dsmil_discover.py`)
  - Kernel module detection (lsmod scanning)
  - Device node enumeration (/dev/)
  - Kernel message analysis (dmesg)
  - PCI/ACPI device detection
  - SMI interface accessibility checks
  - Firmware file discovery
  - System vs framework comparison

- **v1.2.0** (2025-01-06): Interactive menu system
  - Added interactive TUI menu (`dsmil_menu.py`)
  - Device browsing with grouping
  - Device-specific operation menus
  - System status dashboard
  - User-friendly navigation

- **v1.1.0** (2025-01-05): Expanded device integration
  - Added 5 new devices (0x8005, 0x8016, 0x801E, 0x8050, 0x805A)
  - Total: 13 devices integrated (12.0% coverage)
  - New capabilities: Performance monitoring, VPN control, tactical display, storage encryption, sensor fusion
  - Enhanced documentation with new device guides

- **v1.0.0** (2025-01-05): Initial release
  - 8 Priority 1 devices integrated
  - Complete API documentation
  - Full simulation mode
  - Comprehensive testing

## Support

For issues or questions:
1. Check device-specific documentation
2. Review API reference
3. Test in simulation mode first
4. Consult main DSMIL documentation

## Roadmap

**Phase 2:** Additional device integrations
- Group 2 (Network) - 10 more devices
- Group 1 (Extended Security) - 8 more devices

**Phase 3:** Hardware integration
- Real hardware device access
- Kernel module integration
- Performance optimization

---

**Built for Dell Latitude 5450 MIL-SPEC Platform**
**DSMIL Framework - Priority 1: 9 Devices Integrated**
