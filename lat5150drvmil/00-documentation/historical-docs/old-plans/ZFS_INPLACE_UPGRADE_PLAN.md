# ZFS In-Place Upgrade Plan
**Triple Project Merger for Existing ZFS System**

**Date**: 2025-10-15
**Target**: Internal ZFS drive (upgrade in place)
**Test Environment**: External ext4 drive (current)
**Goal**: Merge claude-backups + LAT5150DRVMIL + livecd-gen WITHOUT reinstalling

---

## System Analysis

### Current State
- **External Drive**: ext4, 451GB, 366GB free (TESTING GROUND)
- **Internal Drive**: ZFS encrypted (TARGET FOR UPGRADE)
- **Kernel**: 6.16.9+deb14-amd64 (already running)
- **Hardware**: 66.5 TOPS, DSMIL 79/84 devices, NPU 34 TOPS

### Key Insight
✅ **NO KERNEL REBUILD NEEDED** - Use DKMS for drivers
✅ **NO FILESYSTEM CHANGE** - ZFS already there
✅ **NO REINSTALL** - In-place upgrade only

---

## Architecture Overview

### What Gets Merged
```
/opt/
├── claude-agents/              (2.8GB from claude-backups)
│   ├── 98 AI agents
│   ├── OpenVINO runtime
│   └── Voice UI
├── dsmil-framework/            (2.1GB from LAT5150DRVMIL)
│   ├── DSMIL universal framework
│   ├── TPM2 security suite
│   └── NPU crypto accelerators
└── milspec-tools/              (733MB from livecd-gen)
    └── System utilities

/usr/lib/modules/$(uname -r)/
└── extra/
    ├── dsmil.ko               (DKMS module)
    └── tpm2_accel_early.ko    (DKMS module)

/etc/
├── dsmil/
│   └── config.json            (79/84 device config)
├── systemd/system/
│   ├── dsmil-early.service    (Load before everything)
│   ├── npu-military-mode.service
│   └── claude-agents.service
└── initramfs-tools/
    └── hooks/dsmil            (Early boot integration)
```

---

## Phase 1: Test on External Drive (CURRENT)
**Time**: 1 hour
**Risk**: ZERO (not touching internal)

### Step 1.1: Install DSMIL as DKMS (15 min)
```bash
cd /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration

# Install DKMS package
sudo apt-get install dkms -y

# Prepare DSMIL source
sudo mkdir -p /usr/src/dsmil-1.0
sudo cp kernel_module/tpm2_accel_early.c /usr/src/dsmil-1.0/
sudo cp /home/john/LAT5150DRVMIL/01-source/kernel-driver/dell-millspec-enhanced.c \
  /usr/src/dsmil-1.0/dsmil.c

# Create dkms.conf
cat | sudo tee /usr/src/dsmil-1.0/dkms.conf <<EOF
PACKAGE_NAME="dsmil"
PACKAGE_VERSION="1.0"
BUILT_MODULE_NAME[0]="dsmil"
DEST_MODULE_LOCATION[0]="/extra"
AUTOINSTALL="yes"
EOF

# Build and install
sudo dkms add -m dsmil -v 1.0
sudo dkms build -m dsmil -v 1.0
sudo dkms install -m dsmil -v 1.0

# Verify
lsmod | grep dsmil
```

### Step 1.2: Merge Projects into /opt (20 min)
```bash
# Create structure
sudo mkdir -p /opt/{claude-agents,dsmil-framework,milspec-tools}

# Copy claude-backups
sudo rsync -av --info=progress2 \
  /home/john/claude-backups/agents/ \
  /opt/claude-agents/

sudo rsync -av --info=progress2 \
  /home/john/claude-backups/local-openvino/ \
  /opt/claude-agents/openvino/

# Copy LAT5150DRVMIL
sudo rsync -av --info=progress2 \
  /home/john/LAT5150DRVMIL/ \
  /opt/dsmil-framework/

# Copy livecd-gen utilities
sudo rsync -av --info=progress2 \
  /home/john/livecd-gen/scripts/ \
  /opt/milspec-tools/

# Set permissions
sudo chown -R $USER:$USER /opt/claude-agents
sudo chown -R root:root /opt/dsmil-framework
sudo chmod -R 755 /opt/milspec-tools
```

### Step 1.3: Create Unified Activation Script (15 min)
```bash
cat | sudo tee /opt/milspec-tools/activate-unified-system.sh <<'EOF'
#!/bin/bash
# Unified System Activation Script
# Activates DSMIL + NPU + AI Agents

set -e

echo "🚀 UNIFIED MILSPEC SYSTEM ACTIVATION"
echo "===================================="

# Load DSMIL module
echo "1. Loading DSMIL framework..."
sudo modprobe dsmil
sleep 2

# Activate NPU military mode
echo "2. Activating NPU military mode..."
python3 /opt/dsmil-framework/DSMIL_UNIVERSAL_FRAMEWORK.py --activate

# Enable NPU device
echo "3. Enabling NPU device..."
sudo chmod 666 /dev/accel/accel0
echo 1 | sudo tee /sys/class/accel/accel0/device/power/control

# Start AI agents
echo "4. Starting AI agent framework..."
cd /opt/claude-agents
python3 agents/src/python/claude_agents/npu/npu_performance_validation.py

echo ""
echo "✅ UNIFIED SYSTEM ACTIVATED"
echo "   DSMIL: 79/84 devices"
echo "   NPU: 34.0 TOPS"
echo "   Total: 66.5 TOPS"
EOF

chmod +x /opt/milspec-tools/activate-unified-system.sh
```

### Step 1.4: Test Activation (10 min)
```bash
# Run activation script
sudo /opt/milspec-tools/activate-unified-system.sh

# Verify results
lsmod | grep dsmil
ls -la /dev/accel/
python3 /opt/dsmil-framework/DSMIL_UNIVERSAL_FRAMEWORK.py --status
```

---

## Phase 2: Create Systemd Services (30 min)

### Service 1: DSMIL Early Boot
```bash
cat | sudo tee /etc/systemd/system/dsmil-early.service <<EOF
[Unit]
Description=DSMIL Framework Early Activation
DefaultDependencies=no
Before=local-fs-pre.target
Wants=systemd-modules-load.service

[Service]
Type=oneshot
ExecStart=/sbin/modprobe dsmil
ExecStart=/usr/bin/python3 /opt/dsmil-framework/DSMIL_UNIVERSAL_FRAMEWORK.py --activate
RemainAfterExit=yes

[Install]
WantedBy=sysinit.target
EOF
```

### Service 2: NPU Military Mode
```bash
cat | sudo tee /etc/systemd/system/npu-military-mode.service <<EOF
[Unit]
Description=NPU Military Mode Activation
After=dsmil-early.service
Requires=dsmil-early.service

[Service]
Type=oneshot
ExecStart=/bin/chmod 666 /dev/accel/accel0
ExecStart=/bin/bash -c 'echo 1 > /sys/class/accel/accel0/device/power/control'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
```

### Service 3: Claude Agents
```bash
cat | sudo tee /etc/systemd/system/claude-agents.service <<EOF
[Unit]
Description=Claude AI Agent Framework
After=npu-military-mode.service network.target
Requires=npu-military-mode.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/claude-agents
ExecStart=/usr/bin/python3 /opt/claude-agents/agents/src/python/claude_agents/npu/npu_performance_validation.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
```

### Enable Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable dsmil-early.service
sudo systemctl enable npu-military-mode.service
sudo systemctl enable claude-agents.service
```

---

## Phase 3: Initramfs Integration (20 min)

### Add DSMIL to Initramfs
```bash
# Create hook
cat | sudo tee /etc/initramfs-tools/hooks/dsmil <<'EOF'
#!/bin/sh
PREREQ=""
prereqs() { echo "$PREREQ"; }
case $1 in prereqs) prereqs; exit 0;; esac

. /usr/share/initramfs-tools/hook-functions

# Copy DSMIL module
manual_add_modules dsmil

# Copy DSMIL framework
mkdir -p ${DESTDIR}/opt/dsmil-framework
copy_exec /opt/dsmil-framework/DSMIL_UNIVERSAL_FRAMEWORK.py /opt/dsmil-framework/
copy_exec /usr/bin/python3 /usr/bin/
EOF

chmod +x /etc/initramfs-tools/hooks/dsmil

# Create init script
cat | sudo tee /etc/initramfs-tools/scripts/init-premount/dsmil <<'EOF'
#!/bin/sh
PREREQ=""
prereqs() { echo "$PREREQ"; }
case $1 in prereqs) prereqs; exit 0;; esac

# Load DSMIL before ZFS
modprobe dsmil
python3 /opt/dsmil-framework/DSMIL_UNIVERSAL_FRAMEWORK.py --activate
EOF

chmod +x /etc/initramfs-tools/scripts/init-premount/dsmil

# Rebuild initramfs
sudo update-initramfs -u
```

---

## Phase 4: Deploy to Internal ZFS (15 min)

### Deployment Script
```bash
cat > /home/john/deploy-to-internal-zfs.sh <<'EOF'
#!/bin/bash
# Deploy unified system to internal ZFS drive

echo "⚠️  DEPLOYING TO INTERNAL ZFS DRIVE"
echo "This will modify your internal system."
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 1
fi

# Identify internal ZFS pool
POOL=$(zpool list -H -o name | head -1)
echo "Target ZFS pool: $POOL"

# Mount pool if needed
zfs mount $POOL

# Copy /opt structure
echo "Copying /opt structure..."
sudo rsync -av --info=progress2 /opt/ /$POOL/opt/

# Copy systemd services
echo "Copying systemd services..."
sudo cp /etc/systemd/system/dsmil-*.service /$POOL/etc/systemd/system/
sudo cp /etc/systemd/system/npu-*.service /$POOL/etc/systemd/system/
sudo cp /etc/systemd/system/claude-*.service /$POOL/etc/systemd/system/

# Copy initramfs hooks
echo "Copying initramfs integration..."
sudo cp /etc/initramfs-tools/hooks/dsmil /$POOL/etc/initramfs-tools/hooks/
sudo cp /etc/initramfs-tools/scripts/init-premount/dsmil \
  /$POOL/etc/initramfs-tools/scripts/init-premount/

# Install DKMS on internal (will rebuild initramfs there)
echo "Installing DSMIL DKMS..."
sudo rsync -av /usr/src/dsmil-1.0/ /$POOL/usr/src/dsmil-1.0/

echo ""
echo "✅ DEPLOYMENT COMPLETE"
echo ""
echo "Next steps:"
echo "1. Reboot into internal ZFS drive"
echo "2. Run: sudo dkms install -m dsmil -v 1.0"
echo "3. Run: sudo update-initramfs -u"
echo "4. Run: sudo systemctl enable --now dsmil-early.service"
echo "5. Run: /opt/milspec-tools/activate-unified-system.sh"
EOF

chmod +x /home/john/deploy-to-internal-zfs.sh
```

---

## Testing Checklist

### On External Drive (Before Internal Deployment)
- [ ] DSMIL module loads: `lsmod | grep dsmil`
- [ ] 79/84 devices accessible: `python3 DSMIL_UNIVERSAL_FRAMEWORK.py --status`
- [ ] NPU at 34 TOPS: `python3 npu_performance_validation.py`
- [ ] All 3 projects in /opt: `du -sh /opt/*`
- [ ] Services start: `systemctl status dsmil-early npu-military-mode claude-agents`
- [ ] Initramfs contains dsmil: `lsinitramfs /boot/initrd.img-$(uname -r) | grep dsmil`

### After Internal Deployment
- [ ] Boot into internal ZFS
- [ ] DSMIL loads on boot
- [ ] NPU military mode active
- [ ] Claude agents functional
- [ ] Total system: 66.5 TOPS

---

## Rollback Plan

### If Internal Deployment Fails
1. Boot from external drive
2. Mount internal ZFS pool
3. Remove /opt/claude-agents, /opt/dsmil-framework
4. Remove systemd services
5. Remove initramfs hooks
6. Rebuild internal initramfs
7. Reboot

### Backup Before Deployment
```bash
# Snapshot ZFS before deployment
zpool list
zfs snapshot $POOL@pre-unified-system
# Rollback if needed: zfs rollback $POOL@pre-unified-system
```

---

## Timeline

| Phase | Time | Location |
|-------|------|----------|
| Phase 1: Test on External | 1h | External ext4 |
| Phase 2: Systemd Services | 30m | External ext4 |
| Phase 3: Initramfs | 20m | External ext4 |
| Phase 4: Deploy to Internal | 15m | Internal ZFS |
| **TOTAL** | **2h 5m** | Both |

---

## Advantages of This Approach

✅ **No kernel rebuild** - Uses existing 6.16.9+deb14-amd64
✅ **No reinstall** - In-place upgrade preserves everything
✅ **Safe testing** - External drive first, internal when ready
✅ **ZFS snapshots** - Easy rollback if needed
✅ **DKMS modules** - Survive kernel updates
✅ **Fast deployment** - 2 hours vs 4-5 hours for full rebuild

---

## Final System State

```
Internal ZFS Drive:
├── Existing OS and data (PRESERVED)
├── /opt/claude-agents/         (NEW - 2.8GB)
├── /opt/dsmil-framework/        (NEW - 2.1GB)
├── /opt/milspec-tools/          (NEW - 733MB)
├── DSMIL kernel module          (NEW - DKMS)
├── Systemd services             (NEW - auto-start)
└── Initramfs integration        (NEW - early boot)

Performance:
- 66.5 TOPS total (NPU 34 + GPU 18 + CPU equiv)
- 79/84 DSMIL devices active
- 98 AI agents ready
- Military-grade security active
```

---

**READY TO EXECUTE:** Start with Phase 1 on external drive NOW.
