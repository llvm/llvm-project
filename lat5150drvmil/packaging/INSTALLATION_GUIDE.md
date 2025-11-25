# Dell MIL-SPEC Tools - Installation Guide

## Package Information

**Package Name:** dell-milspec-tools
**Version:** 1.0.0-1
**Architecture:** amd64
**Package File:** dell-milspec-tools_1.0.0-1_amd64.deb
**Size:** 24KB

## Overview

Dell MIL-SPEC Tools provides comprehensive userspace utilities for managing and monitoring Dell Latitude 5450 MIL-SPEC hardware features, including DSMIL device access, TPM2 acceleration, and SMBIOS token management.

## Prerequisites

### System Requirements
- **Platform:** Dell Latitude 5450 MIL-SPEC (recommended)
- **Operating System:** Debian 12+ or Ubuntu 22.04+
- **Kernel:** 6.1.0+
- **Architecture:** x86_64 (amd64)

### Dependencies (Automatically Installed)
- python3 (>= 3.10)
- python3-psutil
- bash (>= 4.4)
- coreutils
- procps

### Recommended (Not Required)
- dell-milspec-dsmil-dkms - Kernel module for DSMIL device
- tpm2-accel-early-dkms - Kernel module for TPM2 acceleration
- sudo - For privileged operations
- dmidecode - For SMBIOS information

## Installation

### Method 1: Using dpkg (Standard)

```bash
# Install the package
sudo dpkg -i dell-milspec-tools_1.0.0-1_amd64.deb

# Install dependencies if missing
sudo apt-get install -f

# Verify installation
dpkg -l | grep dell-milspec-tools
```

### Method 2: Using apt (Recommended)

```bash
# Install using apt (handles dependencies automatically)
sudo apt install ./dell-milspec-tools_1.0.0-1_amd64.deb

# Verify installation
apt list --installed | grep dell-milspec-tools
```

## Post-Installation Steps

### 1. Add User to dsmil Group

During installation, the current user is automatically added to the 'dsmil' group. However, you need to logout and login for the change to take effect.

```bash
# Verify group membership
groups $USER

# If 'dsmil' is not shown, add manually:
sudo usermod -a -G dsmil $USER

# Logout and login to apply changes
```

### 2. Load Kernel Modules

```bash
# Load DSMIL module
sudo modprobe dsmil_72dev

# Load TPM2 acceleration module (optional)
sudo modprobe tpm2_accel_early

# Verify modules are loaded
lsmod | grep -E "(dsmil|tpm2_accel)"
```

### 3. Verify Device Access

```bash
# Check DSMIL device
ls -l /dev/dsmil0

# Check TPM2 device (if applicable)
ls -l /dev/tpm2_accel_early

# Run status check
dsmil-status
```

### 4. Test Basic Functionality

```bash
# Run basic device tests
dsmil-test --basic-only

# Check TPM2 acceleration
tpm2-accel-status

# View system status
milspec-control status
```

## Package Contents

### Installed Commands

| Command | Location | Purpose |
|---------|----------|---------|
| `dsmil-status` | /usr/bin/ | Query DSMIL device status |
| `dsmil-test` | /usr/bin/ | Test DSMIL functionality |
| `tpm2-accel-status` | /usr/bin/ | Query TPM2 acceleration status |
| `milspec-control` | /usr/bin/ | Main control utility |
| `milspec-monitor` | /usr/bin/ | Start monitoring dashboard |
| `milspec-emergency-stop` | /usr/sbin/ | Emergency shutdown procedures |

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `dsmil.conf` | /etc/dell-milspec/ | DSMIL device configuration |
| `monitoring.json` | /etc/dell-milspec/ | Monitoring thresholds |
| `safety.json` | /etc/dell-milspec/ | Safety parameters |

### Monitoring Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `dsmil_comprehensive_monitor.py` | /usr/share/dell-milspec/monitoring/ | Real-time monitoring dashboard |
| `safe_token_tester.py` | /usr/share/dell-milspec/monitoring/ | Safe SMBIOS token testing |

### Examples

| Example | Location | Purpose |
|---------|----------|---------|
| `example-basic-usage.sh` | /usr/share/dell-milspec/examples/ | Basic usage demonstration |
| `example-monitoring.sh` | /usr/share/dell-milspec/examples/ | Monitoring modes demo |
| `example-token-testing.sh` | /usr/share/dell-milspec/examples/ | Token testing demo |

### Directories Created

- `/etc/dell-milspec/` - Configuration files
- `/var/log/dell-milspec/` - Log files
- `/var/run/dell-milspec/` - Runtime data (PID files, etc.)

## Quick Start

### 1. Check System Status

```bash
# View DSMIL device status
dsmil-status

# View TPM2 acceleration status
tpm2-accel-status

# Run comprehensive status check
milspec-control status
```

### 2. Start Monitoring

```bash
# Start dashboard (comprehensive overview)
milspec-monitor

# Or specify a mode
milspec-monitor --mode resources
milspec-monitor --mode tokens
milspec-monitor --mode alerts

# JSON output for scripting
milspec-monitor --json-output
```

### 3. Test Functionality (Safe Mode)

```bash
# Run basic tests
dsmil-test --basic-only

# Run dry-run token tests (no actual modifications)
dsmil-test --dry-run

# Test specific token range
dsmil-test --dry-run --range Range_0480
```

### 4. Use Examples

```bash
# Run basic usage example
bash /usr/share/dell-milspec/examples/example-basic-usage.sh

# Run monitoring example
bash /usr/share/dell-milspec/examples/example-monitoring.sh

# Run token testing example
bash /usr/share/dell-milspec/examples/example-token-testing.sh
```

## Configuration

### DSMIL Configuration

Edit `/etc/dell-milspec/dsmil.conf` to customize:
- Device paths
- Module parameters
- Safety limits
- Logging configuration
- Monitoring settings

```bash
# Edit configuration
sudo nano /etc/dell-milspec/dsmil.conf

# Restart monitoring to apply changes
pkill milspec-monitor
milspec-monitor
```

### Monitoring Thresholds

Edit `/etc/dell-milspec/monitoring.json` to adjust:
- Temperature thresholds (warning, critical, emergency)
- CPU usage limits
- Memory usage limits
- Disk I/O limits
- Token range priorities

```bash
# Edit monitoring configuration
sudo nano /etc/dell-milspec/monitoring.json
```

### Safety Parameters

Edit `/etc/dell-milspec/safety.json` to configure:
- Thermal safety ranges
- Resource safety limits
- Operation limits
- Quarantine rules
- Emergency procedures

```bash
# Edit safety configuration
sudo nano /etc/dell-milspec/safety.json
```

## Troubleshooting

### Module Not Loaded

```bash
# Check if module is available
modinfo dsmil_72dev

# If not found, install DKMS package
sudo apt-get install dell-milspec-dsmil-dkms

# Load module manually
sudo modprobe dsmil_72dev
```

### Device Not Found

```bash
# Check if module created device
ls -l /dev/dsmil0

# Check kernel messages
dmesg | grep dsmil

# Check module parameters
cat /sys/module/dsmil_72dev/parameters/*
```

### Permission Denied

```bash
# Check device permissions
ls -l /dev/dsmil0

# Verify group membership
groups $USER

# Add user to dsmil group if needed
sudo usermod -a -G dsmil $USER
# Logout and login required
```

### High Temperature Warning

```bash
# Check current temperature
cat /sys/class/thermal/thermal_zone*/temp

# Run emergency stop if needed
milspec-emergency-stop

# Allow system to cool before resuming
```

## Uninstallation

### Remove Package (Keep Configuration)

```bash
sudo apt-get remove dell-milspec-tools
```

### Purge Package (Remove Everything)

```bash
sudo apt-get purge dell-milspec-tools
```

### Manual Cleanup

```bash
# Remove configuration
sudo rm -rf /etc/dell-milspec/

# Remove logs
sudo rm -rf /var/log/dell-milspec/

# Remove runtime data
sudo rm -rf /var/run/dell-milspec/

# Remove group
sudo groupdel dsmil
```

## Advanced Usage

### Running as Service

To run monitoring as a systemd service, create:

```bash
sudo nano /etc/systemd/system/milspec-monitor.service
```

```ini
[Unit]
Description=Dell MIL-SPEC Monitoring Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/milspec-monitor --mode dashboard
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable milspec-monitor.service
sudo systemctl start milspec-monitor.service

# Check status
sudo systemctl status milspec-monitor.service
```

### Automated Testing

```bash
# Run tests and save to log
dsmil-test --dry-run --range Range_0480 2>&1 | tee test-$(date +%Y%m%d).log

# Run JSON monitoring for scripting
milspec-monitor --json-output > system-status.json
```

### Emergency Procedures

```bash
# Emergency stop (manual)
milspec-emergency-stop

# Or use control utility
milspec-control emergency

# Or send signal to monitoring process
pkill -TERM milspec-monitor
```

## Support

### Log Files

Check logs for troubleshooting:
- System logs: `/var/log/dell-milspec/system.log`
- Token test logs: `/var/log/dell-milspec/token_test_*.log`
- Emergency logs: `/var/log/dell-milspec/emergency_*.log`

### Diagnostic Information

```bash
# Collect diagnostic information
{
  echo "=== System Information ==="
  uname -a
  cat /etc/os-release

  echo "=== Package Version ==="
  dpkg -l | grep dell-milspec

  echo "=== Loaded Modules ==="
  lsmod | grep -E "(dsmil|tpm2)"

  echo "=== Device Status ==="
  ls -l /dev/dsmil* /dev/tpm2_accel* 2>/dev/null

  echo "=== Recent Logs ==="
  tail -50 /var/log/dell-milspec/system.log

  echo "=== Kernel Messages ==="
  dmesg | grep -i -E "(dsmil|tpm2|dell)" | tail -20
} > diagnostic-report.txt
```

## Safety Notes

1. **Always use dry-run mode first** when testing tokens
2. **Monitor temperature** during operations (target <85°C)
3. **Emergency stop** is always available (Ctrl+C or milspec-emergency-stop)
4. **All operations are logged** for audit trail
5. **Live token testing** may require system restart

## Platform Compatibility

This package is designed for:
- **Primary:** Dell Latitude 5450 MIL-SPEC
- **Compatible:** Other Dell Latitude 5000 series (limited support)
- **Requires:** Intel Core Ultra (Meteor Lake) or similar

## License

GPL-3.0+ - See /usr/share/doc/dell-milspec-tools/copyright

## Version Information

- **Package Version:** 1.0.0-1
- **Release Date:** 2025-10-11
- **Maintainer:** Dell MIL-SPEC Tools Team

---

For additional information, see:
- Examples: `/usr/share/dell-milspec/examples/`
- Configuration: `/etc/dell-milspec/`
- Documentation: `/usr/share/doc/dell-milspec-tools/`
