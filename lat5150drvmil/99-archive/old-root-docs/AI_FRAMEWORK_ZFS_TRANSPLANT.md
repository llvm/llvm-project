# DSMIL AI Framework - ZFS Transplant Guide

**Version:** 8.3.2
**Date:** 2025-10-30
**Target:** Integrate with existing ZFS livecd-xen-ai boot environment

---

## Overview

This guide transplants the complete DSMIL AI Framework into your ZFS encrypted environment alongside the livecd-gen system you just set up.

**What Gets Transplanted:**
- ✅ DSMIL AI Platform (2.1GB source code)
- ✅ Ollama models (~2-11GB)
- ✅ RAG knowledge base
- ✅ Configuration files
- ✅ Chat history (localStorage)
- ✅ All 7 auto-coding tools
- ✅ Complete web interface

**Target ZFS Environment:**
- Pool: `rpool` (encrypted with AES-256-GCM)
- Boot Environment: `livecd-xen-ai`
- Encryption Password: `1/0523/600260`

---

## ZFS Dataset Layout

### Proposed Structure

```
rpool/ai-framework/                (Main dataset)
├── source/                        (Git repository, code)
│   └── mountpoint: /opt/dsmil
│   └── recordsize: 128k
│   └── compression: lz4
│
├── ollama-models/                 (AI models)
│   └── mountpoint: /var/lib/ollama/models
│   └── recordsize: 1M (large files)
│   └── compression: lz4
│
├── rag-index/                     (Knowledge base)
│   └── mountpoint: /var/lib/dsmil/rag
│   └── recordsize: 16k (small files)
│   └── compression: zstd (better ratio)
│
├── config/                        (Configuration)
│   └── mountpoint: /etc/dsmil
│   └── recordsize: 128k
│   └── compression: lz4
│
└── logs/                          (Application logs)
    └── mountpoint: /var/log/dsmil
    └── recordsize: 128k
    └── compression: gzip-9 (maximum)
```

**Benefits:**
- 🔒 All encrypted with rpool password
- 📸 Instant snapshots before changes
- 🗜️ 30-50% space savings via compression
- ✅ Data integrity with checksums
- 🔄 Easy rollback to any snapshot

---

## Prerequisites

### Before Starting

**1. Verify Current AI Framework:**
```bash
cd ~/LAT5150DRVMIL
git status  # Should be clean
ls -la      # Verify all files present
```

**2. Check ZFS Pool Status:**
```bash
# Check if ZFS commands work
zpool status rpool 2>/dev/null || echo "ZFS not available - may be in LiveCD"
```

**3. Know Your Passwords:**
- sudo password: `1786`
- rpool encryption: `1/0523/600260`

**4. Check Disk Space:**
```bash
# AI framework needs ~5-15GB depending on models
zfs list rpool -o name,used,avail
```

---

## Installation Method 1: Automated Script (Recommended)

### One-Command Transplant

```bash
cd ~/LAT5150DRVMIL
./TRANSPLANT_TO_ZFS.sh
```

**The script will:**
1. ✓ Verify prerequisites
2. ✓ Import/unlock ZFS pool
3. ✓ Create safety snapshot
4. ✓ Create optimized ZFS datasets
5. ✓ Copy AI framework source (2.1GB)
6. ✓ Copy Ollama models (if present)
7. ✓ Copy RAG index (if present)
8. ✓ Update systemd service
9. ✓ Update configuration paths
10. ✓ Create integration module for livecd-gen
11. ✓ Verify installation
12. ✓ Create initial snapshots

**Time:** 5-10 minutes (depending on model sizes)

---

## Installation Method 2: Manual Transplant

### Step-by-Step Process

#### Step 1: Create Safety Snapshot

```bash
# Create snapshot of entire pool before changes
sudo zfs snapshot -r rpool@before-ai-transplant-$(date +%Y%m%d)

# Verify
zfs list -t snapshot | grep before-ai-transplant
```

#### Step 2: Create ZFS Datasets

```bash
# Main AI framework dataset
sudo zfs create rpool/ai-framework
sudo zfs set compression=lz4 rpool/ai-framework
sudo zfs set atime=off rpool/ai-framework

# Source code dataset
sudo zfs create rpool/ai-framework/source
sudo zfs set recordsize=128k rpool/ai-framework/source
sudo zfs set mountpoint=/opt/dsmil rpool/ai-framework/source

# Ollama models dataset (large files)
sudo zfs create rpool/ai-framework/ollama-models
sudo zfs set recordsize=1M rpool/ai-framework/ollama-models
sudo zfs set mountpoint=/var/lib/ollama/models rpool/ai-framework/ollama-models

# RAG index dataset (small files, high compression)
sudo zfs create rpool/ai-framework/rag-index
sudo zfs set recordsize=16k rpool/ai-framework/rag-index
sudo zfs set compression=zstd rpool/ai-framework/rag-index
sudo zfs set mountpoint=/var/lib/dsmil/rag rpool/ai-framework/rag-index

# Configuration dataset
sudo zfs create rpool/ai-framework/config
sudo zfs set mountpoint=/etc/dsmil rpool/ai-framework/config

# Logs dataset
sudo zfs create rpool/ai-framework/logs
sudo zfs set recordsize=128k rpool/ai-framework/logs
sudo zfs set compression=gzip-9 rpool/ai-framework/logs
sudo zfs set mountpoint=/var/log/dsmil rpool/ai-framework/logs
```

#### Step 3: Mount Datasets

```bash
# Mount all datasets
sudo zfs mount rpool/ai-framework/source
sudo zfs mount rpool/ai-framework/ollama-models
sudo zfs mount rpool/ai-framework/rag-index
sudo zfs mount rpool/ai-framework/config
sudo zfs mount rpool/ai-framework/logs

# Create parent directories if needed
sudo mkdir -p /opt/dsmil
sudo mkdir -p /var/lib/ollama/models
sudo mkdir -p /var/lib/dsmil/rag
sudo mkdir -p /etc/dsmil
sudo mkdir -p /var/log/dsmil
```

#### Step 4: Copy AI Framework

```bash
# Copy source code
sudo rsync -avh --info=progress2 ~/LAT5150DRVMIL/ /opt/dsmil/

# Copy Ollama models (if present)
if [ -d ~/.ollama ]; then
    sudo rsync -avh --info=progress2 ~/.ollama/ /var/lib/ollama/
fi

# Copy RAG index (if present)
if [ -d ~/.local/share/dsmil ]; then
    sudo rsync -avh --info=progress2 ~/.local/share/dsmil/ /var/lib/dsmil/rag/
fi

# Copy configuration (if present)
if [ -d ~/.config/dsmil ]; then
    sudo cp -r ~/.config/dsmil/* /etc/dsmil/
fi
```

#### Step 5: Set Permissions

```bash
# Set ownership
sudo chown -R $USER:$USER /opt/dsmil
sudo chown -R $USER:$USER /var/lib/dsmil
sudo chown -R $USER:$USER /etc/dsmil
sudo chown -R $USER:$USER /var/log/dsmil

# Ollama user (if ollama is installed)
if id ollama >/dev/null 2>&1; then
    sudo chown -R ollama:ollama /var/lib/ollama
else
    sudo chown -R $USER:$USER /var/lib/ollama
fi
```

#### Step 6: Update Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/dsmil-server.service > /dev/null << 'EOF'
[Unit]
Description=DSMIL Unified AI Server
After=network.target ollama.service zfs-mount.service
Wants=ollama.service
Requires=zfs-mount.service

[Service]
Type=simple
User=john
WorkingDirectory=/opt/dsmil/03-web-interface
ExecStart=/usr/bin/python3 /opt/dsmil/03-web-interface/dsmil_unified_server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
PrivateTmp=true
NoNewPrivileges=true

# Environment
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="DSMIL_HOME=/opt/dsmil"

[Install]
WantedBy=multi-user.target
EOF

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable dsmil-server.service
```

#### Step 7: Update Configuration

```bash
# Update config.json paths
sudo sed -i 's|/home/john/LAT5150DRVMIL|/opt/dsmil|g' /etc/dsmil/config.json
sudo sed -i 's|/home/john/.local/share/dsmil/rag_index|/var/lib/dsmil/rag|g' /etc/dsmil/config.json
```

#### Step 8: Create Initial Snapshots

```bash
# Snapshot each dataset after transplant
sudo zfs snapshot rpool/ai-framework/source@initial-transplant
sudo zfs snapshot rpool/ai-framework/ollama-models@initial-transplant
sudo zfs snapshot rpool/ai-framework/rag-index@initial-transplant
sudo zfs snapshot rpool/ai-framework/config@initial-transplant
sudo zfs snapshot rpool/ai-framework/logs@initial-transplant
```

---

## Integration with livecd-xen-ai Boot Environment

### Add to Boot Environment

**Since you have livecd-xen-ai BE, you can integrate AI framework into it:**

```bash
# Mount the boot environment
sudo zfs set mountpoint=/mnt/livecd-xen-ai rpool/ROOT/livecd-xen-ai
sudo zfs mount rpool/ROOT/livecd-xen-ai

# Copy AI framework into BE
sudo rsync -avh /opt/dsmil/ /mnt/livecd-xen-ai/opt/dsmil/

# Copy systemd service
sudo cp /etc/systemd/system/dsmil-server.service \
        /mnt/livecd-xen-ai/etc/systemd/system/

# Enable service in BE
sudo chroot /mnt/livecd-xen-ai systemctl enable dsmil-server.service

# Unmount
sudo zfs unmount rpool/ROOT/livecd-xen-ai
```

**After reboot into livecd-xen-ai:**
- AI framework will be at `/opt/dsmil`
- Service will auto-start
- All features available

---

## Integration with livecd-gen Build System

### Add AI Framework Module to LiveCD

**Create integration module:**

```bash
# Create module in livecd-gen
cat > ~/livecd-gen/src/modules/dsmil_ai_framework.sh << 'EOF'
#!/bin/bash
# DSMIL AI Framework Integration for LiveCD

install_dsmil_ai_framework() {
    log_info "Installing DSMIL AI Framework from ZFS..."

    # Check if AI framework available
    if [ ! -d "/opt/dsmil" ]; then
        log_warning "AI framework not found, skipping"
        return 0
    fi

    # Copy AI framework to chroot
    rsync -a /opt/dsmil/ "${CHROOT_DIR}/opt/dsmil/"

    # Install Python dependencies
    systemd-nspawn -D "${CHROOT_DIR}" --bind-ro=/etc/resolv.conf \
        /usr/bin/bash -c "
            pip3 install --break-system-packages \
                requests anthropic flask flask-cors \
                beautifulsoup4 sentence-transformers faiss-cpu
        "

    # Copy Ollama models if present
    if [ -d "/var/lib/ollama/models" ]; then
        mkdir -p "${CHROOT_DIR}/var/lib/ollama"
        rsync -a /var/lib/ollama/models/ "${CHROOT_DIR}/var/lib/ollama/models/"
    fi

    # Install systemd service
    if [ -f "/etc/systemd/system/dsmil-server.service" ]; then
        cp /etc/systemd/system/dsmil-server.service \
           "${CHROOT_DIR}/etc/systemd/system/"

        # Enable in chroot
        systemd-nspawn -D "${CHROOT_DIR}" \
            systemctl enable dsmil-server.service
    fi

    log_success "DSMIL AI Framework integrated into LiveCD"
}

# Execute during LiveCD build
install_dsmil_ai_framework
EOF

# Make executable
chmod +x ~/livecd-gen/src/modules/dsmil_ai_framework.sh
```

**Add to build process:**

Edit `~/livecd-gen/build-ultrathink-zfs-native-fixed.sh` and add:
```bash
# After other modules, add:
source "$MODULES_DIR/dsmil_ai_framework.sh"
```

**Now when you build LiveCD:**
```bash
cd ~/livecd-gen
sudo ./build-ultrathink-zfs-native-fixed.sh
```

The resulting ISO will include the complete AI framework!

---

## Verification After Transplant

### Check ZFS Datasets

```bash
# List all AI datasets
zfs list -r rpool/ai-framework

# Check compression ratios
zfs get compressratio rpool/ai-framework -r

# Check space usage
zfs list -o name,used,avail,refer rpool/ai-framework -r
```

### Verify Files

```bash
# Check source code
ls -la /opt/dsmil/
du -sh /opt/dsmil

# Check models
ls -la /var/lib/ollama/models/
du -sh /var/lib/ollama/models/

# Check RAG
ls -la /var/lib/dsmil/rag/
du -sh /var/lib/dsmil/rag/

# Check config
ls -la /etc/dsmil/
cat /etc/dsmil/config.json
```

### Test Service

```bash
# Start service
sudo systemctl start dsmil-server

# Check status
sudo systemctl status dsmil-server

# Test endpoint
curl http://localhost:9876/status

# Check logs
sudo journalctl -u dsmil-server -f
```

### Verify Web Interface

```bash
# Open in browser
xdg-open http://localhost:9876

# Test features:
# - Auto-coding tools should work
# - Chat history should be preserved (localStorage in browser)
# - RAG search should work
# - Model management should list models
```

---

## Snapshot Management

### Create Snapshots

```bash
# Before making changes
sudo zfs snapshot rpool/ai-framework/source@before-update-$(date +%Y%m%d)

# After successful changes
sudo zfs snapshot rpool/ai-framework/source@working-$(date +%Y%m%d)

# Recursive snapshot of all AI datasets
sudo zfs snapshot -r rpool/ai-framework@stable-$(date +%Y%m%d)
```

### List Snapshots

```bash
# List all AI framework snapshots
zfs list -t snapshot | grep ai-framework

# List snapshots for specific dataset
zfs list -t snapshot rpool/ai-framework/source
```

### Rollback to Snapshot

```bash
# Rollback source code only
sudo systemctl stop dsmil-server
sudo zfs rollback rpool/ai-framework/source@working-20251030
sudo systemctl start dsmil-server

# Rollback everything
sudo systemctl stop dsmil-server
sudo zfs rollback -r rpool/ai-framework@stable-20251030
sudo systemctl start dsmil-server
```

---

## Updating AI Framework on ZFS

### Update from Git

```bash
# Stop service
sudo systemctl stop dsmil-server

# Snapshot before update
sudo zfs snapshot rpool/ai-framework/source@before-git-pull-$(date +%Y%m%d)

# Update
cd /opt/dsmil
git pull origin main

# Restart service
sudo systemctl start dsmil-server

# If problems, rollback:
# sudo zfs rollback rpool/ai-framework/source@before-git-pull-*
```

### Update Models

```bash
# Models are in /var/lib/ollama/models (on ZFS)

# Download new model
ollama pull qwen2.5-coder:7b

# Snapshot after download
sudo zfs snapshot rpool/ai-framework/ollama-models@new-model-$(date +%Y%m%d)

# Delete old model
ollama rm qwen2.5-coder:1.5b
```

---

## Integration Scenarios

### Scenario 1: Current System (Before Reboot)

**Install AI framework to current system on ZFS:**

```bash
# Run transplant script
cd ~/LAT5150DRVMIL
./TRANSPLANT_TO_ZFS.sh

# Start service
sudo systemctl start dsmil-server

# Access
xdg-open http://localhost:9876
```

**AI framework now on ZFS, available immediately.**

---

### Scenario 2: After Reboot into livecd-xen-ai

**AI framework already integrated if you:**
1. Ran transplant script before reboot
2. Datasets are shared across boot environments

**After booting into livecd-xen-ai:**

```bash
# Datasets should auto-mount
ls /opt/dsmil  # Should show AI framework
ls /var/lib/ollama/models  # Should show models

# Start service
sudo systemctl start dsmil-server

# Access
xdg-open http://localhost:9876
```

---

### Scenario 3: Include in LiveCD ISO

**Add to livecd-gen build:**

```bash
cd ~/livecd-gen

# Ensure AI framework module exists
ls -la src/modules/dsmil_ai_framework.sh

# Build LiveCD with AI framework
sudo ./build-ultrathink-zfs-native-fixed.sh

# Create ISO
sudo ./create-efi-iso.sh

# Resulting ISO will include complete AI framework
```

---

## Performance Optimization

### ZFS Tuning for AI Workloads

```bash
# Set ARC size for better caching (if you have 32GB+ RAM)
echo "options zfs zfs_arc_max=17179869184" | sudo tee -a /etc/modprobe.d/zfs.conf
# 16GB ARC cache

# Prefetch for RAG dataset (small random reads)
sudo zfs set prefetch=all rpool/ai-framework/rag-index

# Sync for models (large sequential writes)
sudo zfs set sync=disabled rpool/ai-framework/ollama-models

# Apply changes
sudo modprobe -r zfs
sudo modprobe zfs
```

### AI Model Storage Optimization

```bash
# Check compression ratio on models
zfs get compressratio rpool/ai-framework/ollama-models

# If ratio is low (models already compressed), disable compression
sudo zfs set compression=off rpool/ai-framework/ollama-models

# Or try lz4 for better performance
sudo zfs set compression=lz4 rpool/ai-framework/ollama-models
```

---

## Backup and Disaster Recovery

### Backup AI Framework

```bash
# Send to backup drive/server
sudo zfs send -R rpool/ai-framework@stable-20251030 | \
    sudo zfs receive backup-pool/ai-framework-backup

# Or backup to file
sudo zfs send -R rpool/ai-framework@stable-20251030 > \
    /backup/ai-framework-20251030.zfs

# Compress backup
sudo zfs send -R rpool/ai-framework@stable-20251030 | \
    gzip -9 > /backup/ai-framework-20251030.zfs.gz
```

### Restore from Backup

```bash
# From ZFS stream
sudo zfs receive rpool/ai-framework < /backup/ai-framework-20251030.zfs

# From compressed stream
gunzip -c /backup/ai-framework-20251030.zfs.gz | sudo zfs receive rpool/ai-framework
```

---

## Troubleshooting

### Datasets Won't Mount

**Check encryption keys:**
```bash
zfs get keystatus rpool
# If unavailable: echo "1/0523/600260" | sudo zfs load-key rpool
```

**Force mount:**
```bash
sudo zfs mount -a
```

### Service Won't Start

**Check paths:**
```bash
ls -la /opt/dsmil/03-web-interface/dsmil_unified_server.py
# If missing, check if dataset mounted: mount | grep dsmil
```

**Check dependencies:**
```bash
python3 -c "import flask, requests" || pip3 install --user flask requests
```

### Permission Errors

```bash
# Fix ownership
sudo chown -R $USER:$USER /opt/dsmil
sudo chown -R $USER:$USER /var/lib/dsmil
```

---

## Space Management

### Check Space Usage

```bash
# Overall usage
zfs list -r rpool/ai-framework -o name,used,avail,refer

# Detailed per dataset
zfs list -o name,used,refer,compressratio rpool/ai-framework -r

# Snapshots size
zfs list -t snapshot | grep ai-framework | awk '{sum+=$2} END {print sum}'
```

### Clean Old Snapshots

```bash
# List snapshots by age
zfs list -t snapshot -s creation | grep ai-framework

# Delete old snapshots (older than 30 days)
zfs list -t snapshot -o name,creation | grep ai-framework | \
    awk '$2 < "$(date -d "30 days ago" +%s)" {print $1}' | \
    xargs -r -n1 sudo zfs destroy
```

---

## Complete System Layout

After transplant, your system will have:

```
rpool/ (encrypted AES-256-GCM)
├── ROOT/
│   ├── livecd-xen-ai/         (Xen + Kernel 6.16.12)
│   │   └── /opt/dsmil/        → Links to rpool/ai-framework/source
│   │   └── /var/lib/ollama/   → Links to rpool/ai-framework/ollama-models
│   │   └── /etc/dsmil/        → Links to rpool/ai-framework/config
│   └── LONENOMAD_NEW_ROLL/    (Original system, fallback)
│
├── ai-framework/              (NEW - AI Platform)
│   ├── source/                (2.1GB - Code)
│   ├── ollama-models/         (2-11GB - AI models)
│   ├── rag-index/             (100MB-10GB - Knowledge base)
│   ├── config/                (10MB - Configuration)
│   └── logs/                  (10-100MB - Logs)
│
├── home/                      (1.09TB - Your files)
├── datascience/               (13.8GB - ML work)
├── github/                    (28.1GB - Repositories)
└── [other datasets...]
```

---

## Post-Transplant Checklist

After running transplant:

- [ ] ZFS datasets created (5 datasets)
- [ ] AI framework copied to /opt/dsmil
- [ ] Ollama models copied (if present)
- [ ] RAG index copied (if present)
- [ ] Configuration updated
- [ ] Systemd service configured
- [ ] Initial snapshots created
- [ ] Service starts successfully
- [ ] Web interface accessible
- [ ] All features working
- [ ] Integration module created for livecd-gen
- [ ] Original data preserved in ~home/john/LAT5150DRVMIL

---

## Quick Reference

### Transplant Command
```bash
cd ~/LAT5150DRVMIL
./TRANSPLANT_TO_ZFS.sh
```

### Verify Installation
```bash
zfs list -r rpool/ai-framework
sudo systemctl status dsmil-server
curl http://localhost:9876/status
```

### Create Snapshot
```bash
sudo zfs snapshot -r rpool/ai-framework@stable-$(date +%Y%m%d)
```

### Rollback
```bash
sudo systemctl stop dsmil-server
sudo zfs rollback -r rpool/ai-framework@stable-20251030
sudo systemctl start dsmil-server
```

---

**Ready to transplant?** Run `./TRANSPLANT_TO_ZFS.sh` to begin! 🚀
