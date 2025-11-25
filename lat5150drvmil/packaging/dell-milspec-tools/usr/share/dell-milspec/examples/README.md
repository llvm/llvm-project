# Dell MIL-SPEC Tools - Usage Examples

This directory contains example scripts demonstrating the usage of dell-milspec-tools package.

## Available Examples

### 1. Basic Usage Example
**File:** `example-basic-usage.sh`

Demonstrates fundamental operations:
- Checking device status (DSMIL and TPM2)
- Running basic functionality tests
- Viewing configuration
- Starting monitoring dashboard

**Run:** `bash /usr/share/dell-milspec/examples/example-basic-usage.sh`

### 2. Monitoring Example
**File:** `example-monitoring.sh`

Demonstrates different monitoring modes:
- Dashboard mode (comprehensive overview)
- Resource mode (detailed system resources)
- Token mode (DSMIL token monitoring)
- Alert mode (alert history)
- JSON output mode (machine-readable)

**Run:** `bash /usr/share/dell-milspec/examples/example-monitoring.sh`

### 3. Token Testing Example
**File:** `example-token-testing.sh`

Demonstrates safe SMBIOS token testing:
- Testing different token ranges in dry-run mode
- Viewing test logs
- Understanding token range priorities

**Run:** `bash /usr/share/dell-milspec/examples/example-token-testing.sh`

## Quick Start

For first-time users, run the basic usage example:

```bash
bash /usr/share/dell-milspec/examples/example-basic-usage.sh
```

## Prerequisites

Before running examples:

1. Install dell-milspec-tools package:
   ```bash
   sudo dpkg -i dell-milspec-tools_1.0.0-1_amd64.deb
   ```

2. Load kernel modules:
   ```bash
   sudo modprobe dsmil_72dev
   sudo modprobe tpm2_accel_early
   ```

3. Add your user to dsmil group (if not already done):
   ```bash
   sudo usermod -a -G dsmil $USER
   # Logout and login for changes to take effect
   ```

## Safety Notes

All examples use safe defaults:
- Token testing is in DRY RUN mode by default
- Monitoring includes thermal and resource protection
- Emergency stop mechanisms are active
- All operations are logged

## Available Commands

After installation, these commands are available:

- `dsmil-status` - Query DSMIL device status
- `dsmil-test` - Test DSMIL functionality
- `tpm2-accel-status` - Query TPM2 acceleration status
- `milspec-control` - Main control utility
- `milspec-monitor` - Monitoring dashboard
- `milspec-emergency-stop` - Emergency shutdown

## Documentation

Full documentation is available at:
- `/usr/share/doc/dell-milspec-tools/`
- Configuration: `/etc/dell-milspec/`

## Support

For issues or questions:
- Check logs: `/var/log/dell-milspec/`
- Review configuration: `/etc/dell-milspec/`
- Run diagnostics: `dsmil-status`

## Platform

These tools are designed for:
- **Dell Latitude 5450 MIL-SPEC**
- Compatible with dell-milspec-dsmil-dkms kernel module
- Compatible with tpm2-accel-early-dkms kernel module
