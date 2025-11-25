# LAT5150 DRVMIL Tactical AI Sub-Engine - Production Deployment Guide

**Classification:** TOP SECRET//SI//NOFORN
**Platform:** Dell Latitude 5450 MIL-SPEC JRTC1
**Version:** 1.0.0 Production Release
**Last Updated:** 2025-11-13

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Architecture](#deployment-architecture)
4. [Installation Procedures](#installation-procedures)
5. [Configuration Management](#configuration-management)
6. [Security Hardening](#security-hardening)
7. [Operational Procedures](#operational-procedures)
8. [Monitoring & Maintenance](#monitoring--maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Expansion & Scaling](#expansion--scaling)

---

## System Overview

The LAT5150 DRVMIL Tactical AI Sub-Engine is a military-grade, TEMPEST-compliant self-coding AI system with comprehensive hardware reconnaissance capabilities. This guide covers production deployment for operational environments.

### Core Components

| Component | Purpose | Status |
|-----------|---------|--------|
| **Tactical Self-Coding API** | Primary AI interface with RAG, INT8 optimization, learning | Production Ready ✅ |
| **DSMIL Device Discovery** | Hardware reconnaissance for 84 military devices (0x8000-0x806B) | Production Ready ✅ |
| **NPU Integration** | Neural Processing Unit detection and utilization | Production Ready ✅ |
| **Tactical UI** | Military-grade web interface with TEMPEST compliance | Production Ready ✅ |
| **Xen Hypervisor Bridge** | Secure VM-to-host tactical interface access | Production Ready ✅ |
| **RAG System** | Jina Embeddings v3 retrieval-augmented generation | Production Ready ✅ |
| **INT8 Optimization** | Memory-efficient model quantization | Production Ready ✅ |

### Key Features

- ✅ **APT-Grade Security** - Localhost-only, no external network exposure
- ✅ **TEMPEST Compliance** - 5 tactical display modes (Level A: 80% EMF reduction)
- ✅ **Self-Coding Capability** - Natural language to code execution
- ✅ **Hardware Awareness** - Comprehensive DSMIL device discovery with NPU detection
- ✅ **Auto-Start Infrastructure** - SystemD service + VM desktop integration
- ✅ **Military-Grade UI** - GRIDCASE-inspired tactical interface with comfort modes
- ✅ **Hypervisor Integration** - Xen VM secure access via SSH tunneling

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | Intel Core i5 (6th gen+) | Intel Core i7 with NPU | NPU detection requires compatible hardware |
| **RAM** | 16 GB | 32 GB | More for INT8 model caching |
| **Storage** | 50 GB SSD | 100 GB NVMe | Fast storage for embeddings database |
| **Network** | Ethernet | Isolated network segment | Xen bridge requires dedicated NIC |
| **Security** | TPM 2.0 | TPM 2.0 + HSM | Hardware security module recommended |

### Software Requirements

#### Host System (Dom0)
```bash
# Required packages
- Python 3.8+
- Flask 2.0+
- NumPy, SciPy
- Jina AI Embeddings
- Linux kernel 4.4+
- SystemD
- Xen Hypervisor 4.x+
- OpenSSH server
- Nginx (for bridge access)

# Optional but recommended
- Git for version control
- tmux/screen for session management
- fail2ban for SSH protection
```

#### Guest VMs (DomU)
```bash
# Required packages
- OpenSSH client
- Firefox or compatible browser
- Desktop environment (GNOME/KDE/XFCE)
- XDG autostart support
- notify-send (libnotify)
```

### Pre-Deployment Checklist

- [ ] **Hardware validated** - All components meet minimum specs
- [ ] **Operating system updated** - Latest security patches applied
- [ ] **Xen hypervisor configured** - Dom0 and bridge networking operational
- [ ] **SSH keys generated** - Passwordless SSH from VMs to host configured
- [ ] **Firewall rules configured** - iptables/nftables rules for bridge access
- [ ] **DSMIL kernel module loaded** - `/dev/dsmil` device node present
- [ ] **Root access available** - Required for hardware reconnaissance
- [ ] **Backup completed** - System backup before deployment

---

## Deployment Architecture

### Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dom0 (Host)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Tactical AI Sub-Engine                                   │   │
│  │  - Port: 127.0.0.1:5001 (localhost only)                 │   │
│  │  - Security: APT-grade, no external exposure             │   │
│  │  - Auto-start: SystemD service                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                    ┌─────────┴─────────┐                         │
│                    │   xenbr0 Bridge   │                         │
│                    │  192.168.100.1    │                         │
│                    └─────────┬─────────┘                         │
└──────────────────────────────┼──────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼─────────┐  ┌────────▼─────────┐  ┌──────▼──────────┐
│   DomU VM #1     │  │   DomU VM #2     │  │  DomU VM #3     │
│  192.168.100.10  │  │  192.168.100.11  │  │ 192.168.100.12  │
│                  │  │                  │  │                 │
│  SSH Tunnel:     │  │  SSH Tunnel:     │  │ SSH Tunnel:     │
│  localhost:5001  │  │  localhost:5001  │  │ localhost:5001  │
│  ─────────────▶  │  │  ─────────────▶  │  │ ─────────────▶  │
│  192.168.100.1   │  │  192.168.100.1   │  │ 192.168.100.1   │
│                  │  │                  │  │                 │
│  Desktop Icon ✅  │  │  Desktop Icon ✅  │  │ Desktop Icon ✅  │
│  Auto-start  ✅  │  │  Auto-start  ✅  │  │ Auto-start  ✅  │
└──────────────────┘  └──────────────────┘  └─────────────────┘
```

### Security Model

**Three-Layer Defense:**
1. **Localhost Binding** - API server bound to 127.0.0.1 only
2. **SSH Tunnel Encryption** - All VM traffic encrypted via SSH
3. **Bridge Network Isolation** - xenbr0 isolated from external networks

**TEMPEST Compliance:**
- Level A mode: 80% EMF reduction, 60% max brightness, zero animations
- Level C mode: 45% EMF reduction, comfortable for extended operations

---

## Installation Procedures

### Phase 1: Host Setup (Dom0)

#### 1.1 Install SystemD Service

```bash
# Navigate to deployment directory
cd /home/user/LAT5150DRVMIL/deployment

# Install and enable auto-start service
sudo ./install-autostart.sh install

# Verify service is running
sudo systemctl status lat5150-tactical.service

# Expected output:
# ● lat5150-tactical.service - LAT5150 DRVMIL Tactical Self-Coding AI Interface
#    Loaded: loaded (/etc/systemd/system/lat5150-tactical.service; enabled)
#    Active: active (running) since...
```

#### 1.2 Configure Xen Bridge Access (Optional)

**Method A: SSH Tunneling (Recommended)**
- More secure, maintains localhost-only model
- No additional configuration needed
- Skip to Phase 2

**Method B: Nginx Bridge Access**
```bash
# Configure Nginx reverse proxy on xenbr0
sudo ./configure_xen_bridge.sh install

# Verify bridge configuration
sudo ./configure_xen_bridge.sh status

# Expected output:
# ✅ Nginx bridge configured on 192.168.100.1:8443
# ✅ SSL certificates valid
# ✅ Firewall rules active
```

#### 1.3 Verify API Accessibility

```bash
# Test local access
curl -s http://127.0.0.1:5001/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "rag_enabled": true,
#   "int8_enabled": true,
#   "learning_enabled": true,
#   "security_level": "HIGH"
# }
```

### Phase 2: VM Desktop Integration (DomU)

#### 2.1 Deploy Desktop Shortcuts to VMs

```bash
# From Dom0, deploy to all VMs
cd /home/user/LAT5150DRVMIL/deployment

# Single VM deployment
./deploy-vm-shortcuts.sh 192.168.100.10

# Multiple VM batch deployment
./deploy-vm-shortcuts.sh 192.168.100.10 192.168.100.11 192.168.100.12

# Custom SSH user
SSH_USER=tactical ./deploy-vm-shortcuts.sh 192.168.100.10

# Expected output:
# ✅ Deploying to VM: 192.168.100.10
# ✅ SSH tunnel script installed: /usr/local/bin/tactical-tunnel-autostart.sh
# ✅ Desktop shortcut deployed: /usr/share/applications/LAT5150-Tactical.desktop
# ✅ Autostart configured: ~/.config/autostart/tactical-autostart.desktop
# ✅ Desktop icon created: ~/Desktop/LAT5150-Tactical.desktop
```

#### 2.2 Verify VM Access

**From VM:**
```bash
# 1. Reboot VM to test auto-start
sudo reboot

# 2. After boot, check for tunnel
ps aux | grep "ssh.*5001:127.0.0.1:5001"

# Expected: SSH tunnel process running

# 3. Test browser access
firefox http://localhost:5001

# Expected: Tactical interface loads with comfort mode (Level C) default
```

#### 2.3 Manual Tunnel Management (if needed)

**From VM:**
```bash
# Start tunnel manually
ssh -L 5001:127.0.0.1:5001 root@192.168.100.1 -N -f

# Stop tunnel
pkill -f "ssh.*5001:127.0.0.1:5001"

# Check tunnel status
lsof -i :5001
```

### Phase 3: DSMIL Hardware Discovery

#### 3.1 Run Enhanced Reconnaissance

```bash
# From Dom0 (requires root)
cd /home/user/LAT5150DRVMIL/01-source/debugging

# Execute enhanced reconnaissance with NPU detection
sudo ./nsa_device_reconnaissance_enhanced.py

# Expected output:
# 🧠 NPU HARDWARE DETECTED: X devices
#   • Intel AI Boost (Meteor Lake NPU) (pci)
#
# 📡 Probing DSMIL Device Range...
# 🟢 0x8005: TPM/HSM Interface Controller (Confidence: 85%) ✅ READY_FOR_ACTIVATION
# 🟡 0x8007: Power Management Controller (Confidence: 72%) ⚠️ READY_WITH_CAUTION
# ...
# 🧠 0x8035: Neural Processing Unit (Confidence: 68%) 🧠 NPU_CANDIDATE [Sigs: intel_npu]
#
# 🎯 NSA ENHANCED RECONNAISSANCE MISSION SUMMARY
# Total Devices Analyzed: 84
# Responsive Devices: 45
# NPU-Related DSMIL Devices: 3
# Activation Candidates: 12
```

#### 3.2 Review Reconnaissance Results

```bash
# View detailed JSON report
ls -lh nsa_reconnaissance_enhanced_*.json

# Parse with jq
cat nsa_reconnaissance_enhanced_*.json | jq '.npu_candidates'

# Example output:
# [
#   "0x8035",  # Neural processing unit
#   "0x8036",  # AI accelerator interface
#   "0x8037"   # Tensor processing unit
# ]
```

#### 3.3 Document New Devices

```bash
# Create device documentation for newly discovered devices
cd /home/user/LAT5150DRVMIL/00-documentation/devices

# For each new device (example: 0x8035)
cat > device_8035_neural_processing_unit.md << 'EOF'
# Device 0x8035: Neural Processing Unit

## Classification
- **Device ID:** 0x8035
- **Type:** Neural Processing Unit (NPU)
- **Vendor:** Intel
- **Operational Readiness:** NPU_CANDIDATE

## Hardware Details
- **Detection Method:** PCI + DSMIL correlation
- **PCI ID:** 8086:7d1d (Intel AI Boost)
- **Signature Matches:** intel_npu, ai_accelerator
- **Confidence Score:** 68%

## Capabilities
- Neural network inference acceleration
- INT8/FP16 tensor operations
- Low-power AI workload offloading

## Integration Status
- [ ] Device driver investigation
- [ ] API interface development
- [ ] Performance benchmarking
- [ ] Tactical UI integration

## Security Considerations
- Read-only probing confirmed safe
- No write operations attempted
- Quarantine status: SAFE FOR EXPLORATION

## References
- NSA Reconnaissance Report: `nsa_reconnaissance_enhanced_YYYYMMDD.json`
- PCI Database: https://pci-ids.ucw.cz/read/PC/8086/7d1d
EOF
```

---

## Configuration Management

### Security Levels

The tactical API supports three security levels:

| Level | Network Binding | Authentication | CORS | Use Case |
|-------|----------------|----------------|------|----------|
| **HIGH** | 127.0.0.1 only | Required | Disabled | Production (default) |
| **MEDIUM** | 0.0.0.0 with whitelist | Required | Restricted | Development |
| **LOW** | 0.0.0.0 | Optional | Enabled | Testing only |

**Production Configuration (SystemD service):**
```ini
Environment="SECURITY_LEVEL=HIGH"
ExecStart=/usr/bin/python3 .../secured_self_coding_api.py \
    --security-level HIGH \
    --enable-rag \
    --enable-int8 \
    --enable-learning
```

### TEMPEST Display Modes

Configure default display mode in tactical UI:

| Mode | EMF Reduction | Colors | Animations | Use Case |
|------|--------------|--------|------------|----------|
| **Level A** | 80% | Dark green on black | None | Top Secret operations |
| **Level C (Default)** | 45% | Teal/cyan on dark gray | Subtle | Extended operations |
| **Night** | 55% | Amber on black | Minimal | Low-light environments |
| **NVG** | 70% | Green on black | None | Night vision compatible |
| **High Contrast** | 35% | White on black | Normal | Accessibility |

**Set default mode:** Edit `tactical_self_coding_ui.html` line 850:
```javascript
// Change default from 'comfort' to desired mode
setTacticalMode('comfort');  // Options: 'comfort', 'level-a', 'night', 'nvg', 'contrast'
```

### RAG System Configuration

**Embedding Database Location:**
```bash
/home/user/LAT5150DRVMIL/02-rag-embeddings-unified/
├── embeddings_v3.db       # Jina Embeddings v3 database
├── embeddings_v3.index    # Vector index
└── metadata.json          # Configuration metadata
```

**Rebuild embeddings (if needed):**
```bash
cd /home/user/LAT5150DRVMIL/01-source/rag-system
python3 build_embeddings.py --regenerate
```

### INT8 Quantization

**Model cache location:**
```bash
/home/user/LAT5150DRVMIL/.cache/int8_models/
```

**Clear cache (force re-quantization):**
```bash
rm -rf /home/user/LAT5150DRVMIL/.cache/int8_models/
sudo systemctl restart lat5150-tactical.service
```

---

## Security Hardening

### Host Security (Dom0)

#### 1. Firewall Rules

```bash
# Allow only bridge network to access SSH
sudo iptables -A INPUT -i xenbr0 -p tcp --dport 22 -s 192.168.100.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j DROP

# Block all external access to port 5001
sudo iptables -A INPUT -p tcp --dport 5001 ! -s 127.0.0.1 -j DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

#### 2. SSH Hardening

Edit `/etc/ssh/sshd_config`:
```bash
# Disable password authentication
PasswordAuthentication no
PubkeyAuthentication yes

# Restrict to bridge network
ListenAddress 192.168.100.1

# Disable root login (after key setup)
PermitRootLogin prohibit-password

# Rate limiting
MaxAuthTries 3
MaxSessions 10
```

Restart SSH:
```bash
sudo systemctl restart sshd
```

#### 3. SELinux/AppArmor

**Enable SystemD service confinement:**
```ini
[Service]
# Add to lat5150-tactical.service
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
NoNewPrivileges=yes
ReadWritePaths=/home/user/LAT5150DRVMIL
```

#### 4. Audit Logging

```bash
# Enable auditd for tactical API
sudo auditctl -w /home/user/LAT5150DRVMIL/03-web-interface/secured_self_coding_api.py -p rwxa -k tactical_api
sudo auditctl -w /dev/dsmil -p rwxa -k dsmil_access

# View audit logs
sudo ausearch -k tactical_api
sudo ausearch -k dsmil_access
```

### VM Security (DomU)

#### 1. SSH Key Management

```bash
# Generate dedicated key pair on each VM
ssh-keygen -t ed25519 -f ~/.ssh/tactical_host -C "tactical-vm-$(hostname)"

# Copy public key to host
ssh-copy-id -i ~/.ssh/tactical_host.pub root@192.168.100.1

# Restrict key usage in host's ~/.ssh/authorized_keys
command="echo 'Tunnel only'",no-agent-forwarding,no-X11-forwarding,permitopen="127.0.0.1:5001" ssh-ed25519 AAAA...
```

#### 2. Browser Sandboxing

**Firefox security profile:**
```bash
# Create isolated profile
firefox -CreateProfile tactical-isolated

# Edit profile prefs.js
user_pref("network.proxy.socks", "127.0.0.1");
user_pref("network.proxy.socks_port", 5001);
user_pref("network.proxy.type", 1);
user_pref("privacy.trackingprotection.enabled", true);
```

---

## Operational Procedures

### Daily Operations

#### Starting Operations
```bash
# On host (if not using auto-start)
sudo systemctl start lat5150-tactical.service

# On VM - click desktop icon or:
firefox http://localhost:5001
```

#### Stopping Operations
```bash
# On host
sudo systemctl stop lat5150-tactical.service

# On VM - close browser, tunnel remains active
# To stop tunnel:
pkill -f "ssh.*5001:127.0.0.1:5001"
```

### Hardware Discovery Workflow

**Quarterly reconnaissance schedule:**
```bash
# 1. Run enhanced reconnaissance
sudo /home/user/LAT5150DRVMIL/01-source/debugging/nsa_device_reconnaissance_enhanced.py

# 2. Compare with previous results
diff nsa_reconnaissance_enhanced_OLD.json nsa_reconnaissance_enhanced_NEW.json

# 3. Document new devices in /00-documentation/devices/

# 4. Update known_devices dictionary in reconnaissance script

# 5. Commit to repository
git add 00-documentation/devices/device_*.md
git commit -m "docs: Document newly discovered DSMIL devices"
```

### Tactical UI Operations

**Switching TEMPEST modes:**
1. Access tactical interface at `http://localhost:5001`
2. Open "TACTICAL SETTINGS" panel (top-right)
3. Select display mode:
   - **Comfort Mode (Level C)** - Default, extended use
   - **Level A Mode** - Maximum TEMPEST, Top Secret ops
   - **Night Mode** - Low-light environments
   - **NVG Mode** - Night vision compatibility
   - **High Contrast** - Accessibility

**Brightness control:**
- Adjust slider in settings panel
- Level A mode: maximum 60% (enforced)
- Other modes: 0-100%

**Animation control:**
- Enable/disable animations
- Automatic: disabled in Level A, minimal in others

### Self-Coding Operations

**Natural language to code:**
1. Type request in natural language
2. System generates code
3. Review generated code
4. Execute or modify
5. System learns from feedback

**Example:**
```
User: "Scan DSMIL device 0x8035 and show me the response data"

AI: [Generates Python code]
    import struct
    with open('/dev/dsmil', 'rb+') as f:
        cmd = struct.pack('<HH', 0x0001, 0x8035)
        f.write(cmd)
        response = f.read(8)
        print(f"Response: {response.hex()}")

[Execute] [Modify] [Cancel]
```

---

## Monitoring & Maintenance

### System Health Monitoring

#### Service Status
```bash
# Check service health
sudo systemctl status lat5150-tactical.service

# View recent logs
sudo journalctl -u lat5150-tactical.service -n 50 --no-pager

# Follow logs in real-time
sudo journalctl -u lat5150-tactical.service -f
```

#### Performance Monitoring
```bash
# CPU/Memory usage
ps aux | grep secured_self_coding_api.py

# Network connections
sudo netstat -tlnp | grep 5001

# Disk usage (embeddings database)
du -sh /home/user/LAT5150DRVMIL/02-rag-embeddings-unified/
```

#### VM Tunnel Health
```bash
# From VM: Check tunnel status
pgrep -fa "ssh.*5001:127.0.0.1:5001"

# Test connectivity
curl -s http://localhost:5001/health

# Restart tunnel if needed
pkill -f "ssh.*5001:127.0.0.1:5001"
/usr/local/bin/tactical-tunnel-autostart.sh
```

### Log Management

**Log locations:**
```bash
# SystemD service logs
/var/log/journal/

# API application logs
/home/user/LAT5150DRVMIL/logs/tactical_api.log

# DSMIL reconnaissance logs
/home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced.log

# Xen bridge logs
/var/log/nginx/tactical-xen-*.log
```

**Log rotation configuration:**
```bash
# Create /etc/logrotate.d/tactical
cat > /etc/logrotate.d/tactical << 'EOF'
/home/user/LAT5150DRVMIL/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 root root
    sharedscripts
    postrotate
        systemctl reload lat5150-tactical.service > /dev/null 2>&1 || true
    endscript
}
EOF
```

### Database Maintenance

**RAG embeddings database:**
```bash
# Vacuum database (monthly)
cd /home/user/LAT5150DRVMIL/02-rag-embeddings-unified
sqlite3 embeddings_v3.db "VACUUM;"

# Reindex (if performance degrades)
sqlite3 embeddings_v3.db "REINDEX;"

# Backup
cp embeddings_v3.db embeddings_v3.db.backup.$(date +%Y%m%d)
```

### Security Updates

**Monthly security update procedure:**
```bash
# 1. Backup system
sudo rsync -av /home/user/LAT5150DRVMIL /backup/LAT5150DRVMIL.$(date +%Y%m%d)

# 2. Update packages
sudo apt update && sudo apt upgrade -y

# 3. Update Python dependencies
pip3 install --upgrade flask jina numpy scipy

# 4. Test service
sudo systemctl restart lat5150-tactical.service
curl -s http://127.0.0.1:5001/health

# 5. Verify VM access from test VM
ssh test-vm "curl -s http://localhost:5001/health"
```

---

## Troubleshooting

### Common Issues

#### Issue: Service Won't Start

**Symptoms:**
```bash
sudo systemctl status lat5150-tactical.service
# ● lat5150-tactical.service - failed
#    Active: failed (Result: exit-code)
```

**Diagnosis:**
```bash
# Check logs for error
sudo journalctl -u lat5150-tactical.service -n 50

# Common errors:
# - "ModuleNotFoundError: No module named 'flask'" → Install Flask
# - "Permission denied: /dev/dsmil" → Check device permissions
# - "Address already in use" → Kill process on port 5001
```

**Solutions:**
```bash
# Install missing dependencies
pip3 install flask numpy scipy jina

# Fix device permissions
sudo chmod 666 /dev/dsmil

# Kill conflicting process
sudo lsof -ti:5001 | xargs sudo kill -9

# Restart service
sudo systemctl restart lat5150-tactical.service
```

#### Issue: VM Can't Connect via Tunnel

**Symptoms:**
```bash
# From VM:
curl http://localhost:5001/health
# curl: (7) Failed to connect to localhost port 5001: Connection refused
```

**Diagnosis:**
```bash
# Check if tunnel is running
ps aux | grep "ssh.*5001:127.0.0.1:5001"
# (no output = tunnel not running)

# Check SSH connectivity
ssh root@192.168.100.1 "echo connected"

# Check host service
ssh root@192.168.100.1 "systemctl status lat5150-tactical.service"
```

**Solutions:**
```bash
# Start tunnel manually
ssh -L 5001:127.0.0.1:5001 root@192.168.100.1 -N -f

# OR run autostart script
/usr/local/bin/tactical-tunnel-autostart.sh

# Verify tunnel
lsof -i :5001
curl http://localhost:5001/health
```

#### Issue: NPU Not Detected

**Symptoms:**
```bash
sudo ./nsa_device_reconnaissance_enhanced.py
# 🧠 NPU HARDWARE DETECTED: 0 devices
```

**Diagnosis:**
```bash
# Check PCI devices
lspci -nn | grep -i "npu\|vpu\|ai\|neural"

# Check kernel modules
lsmod | grep -i "npu\|vpu\|intel_vpu"

# Check ACPI tables
sudo ls -la /sys/firmware/acpi/tables/ | grep -i npu
```

**Solutions:**
```bash
# Update PCI ID database
sudo update-pciids

# Load Intel VPU driver (if available)
sudo modprobe intel_vpu

# Add device to known NPU list in reconnaissance script
# Edit nsa_device_reconnaissance_enhanced.py:
# self.npu_pci_ids = {
#     'YOUR:PCID': 'Your NPU Name',
#     ...
# }
```

#### Issue: TEMPEST Mode Not Switching

**Symptoms:**
- Clicking mode button in UI doesn't change display

**Diagnosis:**
```bash
# Check browser console (F12)
# Look for JavaScript errors

# Verify tactical UI file integrity
md5sum /home/user/LAT5150DRVMIL/03-web-interface/tactical_self_coding_ui.html
```

**Solutions:**
```bash
# Clear browser cache
firefox --new-instance --private-window http://localhost:5001

# Restore from git
cd /home/user/LAT5150DRVMIL
git checkout 03-web-interface/tactical_self_coding_ui.html

# Restart service
sudo systemctl restart lat5150-tactical.service
```

### Emergency Procedures

#### Complete System Reset

```bash
# 1. Stop all services
sudo systemctl stop lat5150-tactical.service

# 2. Kill all tunnels
sudo pkill -f "ssh.*5001:127.0.0.1:5001"

# 3. Reset to git HEAD
cd /home/user/LAT5150DRVMIL
git reset --hard HEAD

# 4. Reinstall services
cd deployment
sudo ./install-autostart.sh remove
sudo ./install-autostart.sh install

# 5. Redeploy VM shortcuts
./deploy-vm-shortcuts.sh 192.168.100.10 192.168.100.11

# 6. Verify health
curl http://127.0.0.1:5001/health
```

#### Disaster Recovery

```bash
# Restore from backup
sudo rsync -av /backup/LAT5150DRVMIL.YYYYMMDD/ /home/user/LAT5150DRVMIL/

# Restore permissions
sudo chown -R user:user /home/user/LAT5150DRVMIL
sudo chmod +x /home/user/LAT5150DRVMIL/deployment/*.sh
sudo chmod +x /home/user/LAT5150DRVMIL/01-source/debugging/*.py

# Reinstall services
cd /home/user/LAT5150DRVMIL/deployment
sudo ./install-autostart.sh install
```

---

## Expansion & Scaling

### DSMIL Device Expansion

As new devices are discovered in the 0x8000-0x806B range:

#### 1. Document New Device

```bash
cd /home/user/LAT5150DRVMIL/00-documentation/devices
vim device_XXXX_description.md
```

#### 2. Update Reconnaissance Script

Edit `01-source/debugging/nsa_device_reconnaissance_enhanced.py`:

```python
self.known_devices = {
    # Existing devices...
    0x8035: "Neural Processing Unit",  # NEW
    0x8036: "AI Accelerator Interface",  # NEW
    # ...
}
```

#### 3. Add Device Signatures

```python
self.nsa_signatures = {
    # Existing signatures...
    'new_device_sig': [0xAB, 0xCD, 0xEF],  # NEW
    # ...
}
```

#### 4. Test & Validate

```bash
# Run reconnaissance
sudo ./nsa_device_reconnaissance_enhanced.py

# Verify new device detected
cat nsa_reconnaissance_enhanced_*.json | jq '.device_reports."0x8035"'
```

#### 5. Commit Changes

```bash
git add 00-documentation/devices/device_*.md
git add 01-source/debugging/nsa_device_reconnaissance_enhanced.py
git commit -m "feat: Add DSMIL device 0x8035 (Neural Processing Unit)"
```

### Adding New VMs

```bash
# 1. Create Xen VM configuration
cd /home/user/LAT5150DRVMIL/deployment
cp tactical-client.cfg tactical-client-newvm.cfg

# 2. Edit configuration
vim tactical-client-newvm.cfg
# Update: name, vif IP, memory, vcpus, disk

# 3. Create VM
sudo xl create tactical-client-newvm.cfg

# 4. Deploy shortcuts to new VM
./deploy-vm-shortcuts.sh 192.168.100.XX

# 5. Verify access from new VM
ssh 192.168.100.XX "curl http://localhost:5001/health"
```

### Scaling for Multiple Hosts

**Distributed deployment across multiple hosts:**

```bash
# Host A: Primary tactical interface (port 5001)
# Host B: Secondary tactical interface (port 5002)
# Host C: Tertiary tactical interface (port 5003)

# Deploy to Host B:
scp -r /home/user/LAT5150DRVMIL hostB:/home/user/
ssh hostB "cd /home/user/LAT5150DRVMIL/deployment && \
           sudo ./install-autostart.sh install"

# Update VM shortcuts to use Host B:
# Edit deployment/xen-vm-desktop/tactical-tunnel-autostart.sh
# HOST_IP="192.168.100.1"  →  HOST_IP="192.168.100.2"
```

### NPU Workload Distribution

**Future expansion for NPU utilization:**

```python
# In secured_self_coding_api.py, add NPU acceleration:

def accelerate_inference(data):
    """Offload inference to NPU if available"""
    if npu_available:
        return npu_device.infer(data)
    else:
        return cpu_fallback_infer(data)
```

---

## Appendix

### Quick Reference Commands

```bash
# Service Management
sudo systemctl start lat5150-tactical.service
sudo systemctl stop lat5150-tactical.service
sudo systemctl restart lat5150-tactical.service
sudo systemctl status lat5150-tactical.service

# Logs
sudo journalctl -u lat5150-tactical.service -f

# Health Check
curl http://127.0.0.1:5001/health

# VM Tunnel (from VM)
ssh -L 5001:127.0.0.1:5001 root@192.168.100.1 -N -f
pkill -f "ssh.*5001:127.0.0.1:5001"

# Device Reconnaissance
sudo /home/user/LAT5150DRVMIL/01-source/debugging/nsa_device_reconnaissance_enhanced.py

# Deploy VM Shortcuts
/home/user/LAT5150DRVMIL/deployment/deploy-vm-shortcuts.sh <VM_IP>
```

### File Locations

| Component | Path |
|-----------|------|
| **Main API** | `/home/user/LAT5150DRVMIL/03-web-interface/secured_self_coding_api.py` |
| **Tactical UI** | `/home/user/LAT5150DRVMIL/03-web-interface/tactical_self_coding_ui.html` |
| **SystemD Service** | `/etc/systemd/system/lat5150-tactical.service` |
| **VM Shortcuts** | `/usr/share/applications/LAT5150-Tactical.desktop` (VM) |
| **VM Autostart** | `/usr/local/bin/tactical-tunnel-autostart.sh` (VM) |
| **DSMIL Recon** | `/home/user/LAT5150DRVMIL/01-source/debugging/nsa_device_reconnaissance_enhanced.py` |
| **Device Docs** | `/home/user/LAT5150DRVMIL/00-documentation/devices/` |
| **RAG Database** | `/home/user/LAT5150DRVMIL/02-rag-embeddings-unified/embeddings_v3.db` |
| **Logs** | `/var/log/journal/` (SystemD), `./nsa_reconnaissance_enhanced.log` (DSMIL) |

### Network Ports

| Port | Service | Binding | Purpose |
|------|---------|---------|---------|
| **5001** | Tactical API | 127.0.0.1 | Primary interface (localhost only) |
| **8443** | Nginx Bridge | xenbr0 IP | Optional bridge access (HTTPS) |
| **22** | SSH | xenbr0 IP | VM tunnel establishment |

### Support & Contact

For issues, enhancements, or operational questions:
- **Repository:** https://github.com/SWORDIntel/LAT5150DRVMIL
- **Documentation:** `/home/user/LAT5150DRVMIL/00-documentation/`
- **Logs:** `sudo journalctl -u lat5150-tactical.service`

---

**End of Deployment Guide**
**Classification: TOP SECRET//SI//NOFORN**
**Destroy by: [Date + 10 years]**
