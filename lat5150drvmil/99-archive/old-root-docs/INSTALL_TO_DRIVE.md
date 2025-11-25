# DSMIL Platform - Install to Custom Drive or ZFS Pool

**Version:** 8.3.2
**Purpose:** Install or transplant DSMIL platform to a different drive (local disk, ZFS pool, etc.)

---

## Overview

This guide helps you install the DSMIL platform to a location other than your home directory:
- Different local drive (SSD, HDD, NVMe)
- ZFS pool (with snapshots and compression)
- Network storage (NFS, CIFS) - not recommended for performance
- External drive (USB, Thunderbolt)

**Benefits:**
- ✅ More disk space available
- ✅ Better I/O performance (dedicated drive)
- ✅ ZFS snapshots for rollback
- ✅ Separation from home directory
- ✅ Easier backup and management

---

## Pre-Installation Planning

### 1. Choose Target Location

**Check available drives:**
```bash
# List all mounted filesystems
df -h

# List all block devices
lsblk

# Check ZFS pools (if using ZFS)
zfs list
```

**Recommended locations:**
- `/opt/dsmil` - Traditional location for optional software
- `/srv/dsmil` - Service data location
- `/data/dsmil` - Custom data drive
- `/mnt/storage/dsmil` - Mounted external drive
- `<zfs-pool>/dsmil` - ZFS dataset

### 2. Check Disk Space Requirements

```bash
# Check current DSMIL size
du -sh ~/LAT5150DRVMIL

# Check target drive space
df -h /target/path
```

**Space requirements:**
- Source code: ~50MB
- AI models: 2-11GB
- RAG index: 100MB-10GB (grows with use)
- Logs: 10-100MB
- **Total:** 5-25GB recommended

---

## Installation Methods

### Method 1: Fresh Installation to Custom Drive

**Install directly to target location:**

```bash
# 1. Choose target directory
TARGET_DIR="/opt/dsmil"  # Or /data/dsmil, /mnt/storage/dsmil, etc.

# 2. Create directory (may need sudo)
sudo mkdir -p $TARGET_DIR
sudo chown $USER:$USER $TARGET_DIR

# 3. Clone repository to target
cd $TARGET_DIR
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .

# 4. Run installer
./install-complete.sh

# 5. Update systemd service to use new location
sudo sed -i "s|/home/john/LAT5150DRVMIL|$TARGET_DIR|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

---

### Method 2: Transplant Existing Installation

**Move existing installation to new drive:**

```bash
# 1. Stop service
sudo systemctl stop dsmil-server

# 2. Choose target
TARGET_DIR="/opt/dsmil"

# 3. Create directory
sudo mkdir -p $TARGET_DIR
sudo chown $USER:$USER $TARGET_DIR

# 4. Copy installation
rsync -avh ~/LAT5150DRVMIL/ $TARGET_DIR/

# 5. Verify copy
du -sh ~/LAT5150DRVMIL $TARGET_DIR

# 6. Update systemd service
sudo sed -i "s|$HOME/LAT5150DRVMIL|$TARGET_DIR|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload

# 7. Update configuration
sed -i "s|$HOME/LAT5150DRVMIL|$TARGET_DIR|g" ~/.config/dsmil/config.json

# 8. Restart service
sudo systemctl start dsmil-server

# 9. Verify it works
curl http://localhost:9876/status

# 10. Remove old installation (optional)
rm -rf ~/LAT5150DRVMIL
```

---

## ZFS-Specific Installation

### Benefits of ZFS

- **Snapshots** - Instant backup before changes
- **Compression** - Save 30-50% disk space
- **Checksums** - Data integrity verification
- **Clones** - Test changes without duplicating data
- **Rollback** - Instantly revert to previous state

### Create ZFS Dataset

```bash
# Assuming you have a ZFS pool named 'tank'
# List pools
zpool list

# Create dataset for DSMIL
sudo zfs create tank/dsmil

# Set properties
sudo zfs set compression=lz4 tank/dsmil       # Enable compression
sudo zfs set atime=off tank/dsmil             # Disable access time (performance)
sudo zfs set recordsize=128k tank/dsmil       # Optimize for large files
sudo zfs set snapdir=visible tank/dsmil       # Make snapshots visible

# Set ownership
sudo chown -R $USER:$USER /tank/dsmil

# Verify
zfs get all tank/dsmil | grep -E "compression|atime|recordsize"
```

### Install to ZFS Dataset

```bash
# Clone to ZFS dataset
cd /tank/dsmil
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .

# Run installer
./install-complete.sh

# Create initial snapshot
sudo zfs snapshot tank/dsmil@initial-install

# Update systemd to use ZFS location
sudo sed -i "s|/home/john/LAT5150DRVMIL|/tank/dsmil|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

### ZFS Snapshots for Safety

**Create snapshot before changes:**
```bash
# Before updating or testing changes
sudo zfs snapshot tank/dsmil@before-update-$(date +%Y%m%d)

# List snapshots
zfs list -t snapshot | grep dsmil

# Rollback if something breaks
sudo zfs rollback tank/dsmil@before-update-20251030
sudo systemctl restart dsmil-server
```

**Automatic snapshots:**
```bash
# Create pre-update script
cat > /tank/dsmil/pre-update.sh << 'EOF'
#!/bin/bash
sudo zfs snapshot tank/dsmil@auto-$(date +%Y%m%d-%H%M%S)
echo "Snapshot created: tank/dsmil@auto-$(date +%Y%m%d-%H%M%S)"
EOF

chmod +x /tank/dsmil/pre-update.sh

# Run before updates
./pre-update.sh
git pull
sudo systemctl restart dsmil-server
```

---

## Optimizations for Different Storage Types

### SSD Optimization

```bash
# Disable access time updates (performance)
sudo mount -o remount,noatime /mnt/ssd

# Or in /etc/fstab:
/dev/sdX  /mnt/ssd  ext4  noatime,discard  0  2
```

### NVMe Optimization

```bash
# NVMe drives are already fast
# Ensure proper alignment
sudo parted /dev/nvmeXnY align-check optimal 1

# Check NVMe queue depth
cat /sys/block/nvme0n1/queue/nr_requests
```

### ZFS Optimization (Detailed)

```bash
# Create optimized ZFS dataset for DSMIL
sudo zfs create tank/dsmil

# AI model storage (large files)
sudo zfs set compression=lz4 tank/dsmil
sudo zfs set recordsize=1M tank/dsmil/models     # Large sequential files
sudo zfs set primarycache=metadata tank/dsmil/models  # Save RAM for AI

# RAG index storage (small files, lots of random access)
sudo zfs create tank/dsmil/rag
sudo zfs set recordsize=16k tank/dsmil/rag       # Small files
sudo zfs set compression=zstd tank/dsmil/rag     # Better compression
sudo zfs set primarycache=all tank/dsmil/rag     # Cache everything

# Logs (sequential writes)
sudo zfs create tank/dsmil/logs
sudo zfs set recordsize=128k tank/dsmil/logs
sudo zfs set compression=gzip tank/dsmil/logs    # High compression for logs
sudo zfs set sync=disabled tank/dsmil/logs       # Async writes (faster)

# Set ownership
sudo chown -R $USER:$USER /tank/dsmil
```

### HDD Optimization

```bash
# Use ext4 with optimizations
sudo mkfs.ext4 -m 1 -O ^has_journal /dev/sdX

# Mount with optimizations
sudo mount -o noatime,data=writeback /dev/sdX /mnt/hdd
```

---

## Symlink Configuration

### Keep Source on One Drive, Data on Another

**Example: Source on SSD, models/data on HDD**

```bash
# Install to SSD
cd /ssd/dsmil
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .

# Move large data to HDD
mkdir -p /hdd/dsmil-data/models
mkdir -p /hdd/dsmil-data/rag
mkdir -p /hdd/dsmil-data/logs

# Create symlinks
ln -s /hdd/dsmil-data/models ~/.ollama
ln -s /hdd/dsmil-data/rag ~/.local/share/dsmil/rag_index
ln -s /hdd/dsmil-data/logs /ssd/dsmil/logs

# Update config
nano ~/.config/dsmil/config.json
# Set paths to symlink locations
```

---

## Multi-User Installation

### System-Wide Installation

**Install for all users on the system:**

```bash
# Install to /opt (system-wide)
sudo mkdir -p /opt/dsmil
sudo chown root:users /opt/dsmil
sudo chmod 775 /opt/dsmil

cd /opt/dsmil
sudo -u $USER git clone https://github.com/SWORDIntel/LAT5150DRVMIL .

# Run installer
sudo -u $USER ./install-complete.sh

# Each user gets their own config
# User 1: ~/.config/dsmil/config.json
# User 2: ~/.config/dsmil/config.json
# etc.

# But shares the source code and models
```

---

## Transplant with Preservation

### Move Installation While Preserving Data

**Preserve chat history, RAG index, and configuration:**

```bash
# 1. Stop service
sudo systemctl stop dsmil-server

# 2. Backup user data
mkdir -p ~/dsmil-backup
cp -r ~/.config/dsmil ~/dsmil-backup/config
cp -r ~/.local/share/dsmil ~/dsmil-backup/data

# 3. Move source code
TARGET="/opt/dsmil"
sudo mkdir -p $TARGET
sudo chown $USER:$USER $TARGET
rsync -avh ~/LAT5150DRVMIL/ $TARGET/

# 4. Update paths in systemd
sudo sed -i "s|$HOME/LAT5150DRVMIL|$TARGET|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload

# 5. Update config
sed -i "s|$HOME/LAT5150DRVMIL|$TARGET|g" ~/.config/dsmil/config.json

# 6. Start service
sudo systemctl start dsmil-server

# 7. Verify chat history preserved
curl http://localhost:9876
# Check browser localStorage - chats should still be there

# 8. Remove old source (keep backup for now)
mv ~/LAT5150DRVMIL ~/LAT5150DRVMIL.old
```

---

## ZFS Transplant Guide

### Complete ZFS Migration

**Move existing installation to ZFS with full optimization:**

```bash
# 1. Create ZFS dataset
sudo zfs create tank/dsmil
sudo zfs set compression=lz4 tank/dsmil
sudo zfs set atime=off tank/dsmil
sudo chown $USER:$USER /tank/dsmil

# 2. Stop service
sudo systemctl stop dsmil-server

# 3. Take snapshot of current state
tar -czf ~/dsmil-backup-$(date +%Y%m%d).tar.gz \
    ~/LAT5150DRVMIL \
    ~/.config/dsmil \
    ~/.local/share/dsmil

# 4. Copy to ZFS
rsync -avhP ~/LAT5150DRVMIL/ /tank/dsmil/

# 5. Move Ollama models to ZFS (saves space)
sudo systemctl stop ollama

sudo zfs create tank/dsmil/ollama-models
sudo zfs set recordsize=1M tank/dsmil/ollama-models
sudo chown $USER:$USER /tank/dsmil/ollama-models

# Move models
mv ~/.ollama/models/* /tank/dsmil/ollama-models/
rmdir ~/.ollama/models
ln -s /tank/dsmil/ollama-models ~/.ollama/models

sudo systemctl start ollama

# 6. Move RAG index to ZFS
sudo zfs create tank/dsmil/rag-index
sudo zfs set recordsize=16k tank/dsmil/rag-index
sudo zfs set compression=zstd tank/dsmil/rag-index
sudo chown $USER:$USER /tank/dsmil/rag-index

mv ~/.local/share/dsmil/rag_index/* /tank/dsmil/rag-index/
rmdir ~/.local/share/dsmil/rag_index
ln -s /tank/dsmil/rag-index ~/.local/share/dsmil/rag_index

# 7. Update systemd service
sudo sed -i "s|$HOME/LAT5150DRVMIL|/tank/dsmil|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload

# 8. Update config
sed -i 's|"install_dir": ".*"|"install_dir": "/tank/dsmil"|g' ~/.config/dsmil/config.json
sed -i 's|"index_dir": ".*"|"index_dir": "/tank/dsmil/rag-index"|g' ~/.config/dsmil/config.json

# 9. Create initial snapshot
sudo zfs snapshot tank/dsmil@transplant-complete-$(date +%Y%m%d)
sudo zfs snapshot tank/dsmil/ollama-models@initial
sudo zfs snapshot tank/dsmil/rag-index@initial

# 10. Start service
sudo systemctl start dsmil-server

# 11. Verify
curl http://localhost:9876/status
ollama list
ls /tank/dsmil/

# 12. Test for a day, then remove old installation
# rm -rf ~/LAT5150DRVMIL.old
```

---

## ZFS Layout (Recommended Structure)

```
tank/dsmil                      (Main dataset)
├── @transplant-complete-20251030 (snapshot)
├── source/                     (Git repository)
├── ollama-models/              (ZFS dataset, 1M recordsize)
│   └── @initial                (snapshot)
├── rag-index/                  (ZFS dataset, 16k recordsize, zstd)
│   └── @initial                (snapshot)
└── logs/                       (ZFS dataset, 128k recordsize, gzip)
    └── @daily-YYYYMMDD         (auto snapshots)
```

**Create this layout:**
```bash
sudo zfs create tank/dsmil
sudo zfs create tank/dsmil/source
sudo zfs create tank/dsmil/ollama-models
sudo zfs create tank/dsmil/rag-index
sudo zfs create tank/dsmil/logs

# Set properties
sudo zfs set compression=lz4 tank/dsmil/source
sudo zfs set recordsize=1M tank/dsmil/ollama-models
sudo zfs set recordsize=16k compression=zstd tank/dsmil/rag-index
sudo zfs set recordsize=128k compression=gzip tank/dsmil/logs

# Set ownership
sudo chown -R $USER:$USER /tank/dsmil

# Clone to source directory
cd /tank/dsmil/source
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .
```

---

## Automated ZFS Snapshots

### Daily Snapshots

```bash
# Create snapshot script
sudo tee /usr/local/bin/dsmil-snapshot.sh > /dev/null << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M)

zfs snapshot tank/dsmil@daily-$DATE
zfs snapshot tank/dsmil/rag-index@daily-$DATE

# Keep only last 7 days of daily snapshots
zfs list -t snapshot -o name | grep tank/dsmil@daily | head -n -7 | xargs -r -n1 zfs destroy

echo "Snapshots created: tank/dsmil@daily-$DATE"
EOF

sudo chmod +x /usr/local/bin/dsmil-snapshot.sh

# Add to cron (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/dsmil-snapshot.sh") | crontab -
```

### Pre-Update Snapshots

```bash
# Create pre-update hook
cat > /tank/dsmil/source/pre-update-hook.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
sudo zfs snapshot tank/dsmil@pre-update-$DATE
echo "Snapshot created: tank/dsmil@pre-update-$DATE"
echo "To rollback: sudo zfs rollback tank/dsmil@pre-update-$DATE"
EOF

chmod +x /tank/dsmil/source/pre-update-hook.sh

# Use before updates
cd /tank/dsmil/source
./pre-update-hook.sh
git pull
sudo systemctl restart dsmil-server
```

---

## Rollback Procedures

### Git Rollback (Code Only)

```bash
cd /tank/dsmil/source

# View commit history
git log --oneline -10

# Rollback to previous commit
git reset --hard HEAD~1

# Restart service
sudo systemctl restart dsmil-server
```

### ZFS Rollback (Complete State)

```bash
# List snapshots
zfs list -t snapshot | grep dsmil

# Rollback to snapshot (destroys newer snapshots!)
sudo systemctl stop dsmil-server
sudo zfs rollback tank/dsmil@before-update-20251030
sudo systemctl start dsmil-server

# Rollback to snapshot (keep newer snapshots)
sudo zfs rollback -r tank/dsmil@before-update-20251030
```

### ZFS Clone (Test Without Risk)

```bash
# Create clone for testing
sudo zfs clone tank/dsmil@initial tank/dsmil-test

# Test changes on clone
cd /tank/dsmil-test/source
# Make changes, test, etc.

# If good, promote clone to main
sudo zfs promote tank/dsmil-test
sudo zfs rename tank/dsmil tank/dsmil-old
sudo zfs rename tank/dsmil-test tank/dsmil

# If bad, just destroy clone
sudo zfs destroy tank/dsmil-test
```

---

## Performance Comparison

### Different Storage Types

| Storage | Read | Write | Best For |
|---------|------|-------|----------|
| **NVMe SSD** | 3000+ MB/s | 2000+ MB/s | Source code, active development |
| **SATA SSD** | 500 MB/s | 400 MB/s | General use |
| **HDD** | 150 MB/s | 150 MB/s | Model storage, archives |
| **ZFS (SSD)** | 2500 MB/s | 1800 MB/s | Best overall (snapshots + speed) |
| **ZFS (HDD)** | 140 MB/s | 120 MB/s | Large storage with protection |
| **USB 3.0** | 100 MB/s | 50 MB/s | Portable, not for performance |

### Recommended Layout

**Option 1: All on NVMe (Best Performance)**
```
/nvme/dsmil/
├── source/         (Git repo)
├── models/         (Ollama models)
└── data/           (RAG, logs)
```

**Option 2: Split SSD/HDD (Balanced)**
```
/ssd/dsmil/         (Source code, active data)
/hdd/dsmil-storage/ (Models, archives)
  ├── models/       (Symlink from ~/.ollama/models)
  └── archives/     (Old data)
```

**Option 3: ZFS Pool (Best Reliability)**
```
tank/dsmil/
├── source/         (ZFS dataset, lz4)
├── ollama-models/  (ZFS dataset, 1M recordsize)
├── rag-index/      (ZFS dataset, 16k, zstd)
└── logs/           (ZFS dataset, 128k, gzip)
```

---

## Configuration Updates

### Update All Path References

After moving to new location, update these files:

**1. Systemd service:**
```bash
sudo nano /etc/systemd/system/dsmil-server.service

# Update WorkingDirectory and ExecStart:
WorkingDirectory=/new/path/03-web-interface
ExecStart=/usr/bin/python3 /new/path/03-web-interface/dsmil_unified_server.py
```

**2. User configuration:**
```bash
nano ~/.config/dsmil/config.json

# Update:
"install_dir": "/new/path"
"index_dir": "/new/path/rag-index"
```

**3. Environment variables (if used):**
```bash
nano ~/.bashrc

# Update any DSMIL_HOME or similar variables
export DSMIL_HOME="/new/path"
```

**4. Reload and restart:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
source ~/.bashrc
```

---

## Verification

### Verify New Installation

```bash
# Check service is using new location
sudo systemctl status dsmil-server | grep -i working

# Check config
cat ~/.config/dsmil/config.json | grep install_dir

# Test endpoint
curl http://localhost:9876/status

# Check disk usage on new drive
df -h /new/path

# Verify models accessible
ollama list

# Test RAG
curl "http://localhost:9876/rag/stats"
```

### ZFS Verification

```bash
# Check ZFS dataset usage
zfs list -o name,used,avail,refer tank/dsmil

# Check compression ratio
zfs get compressratio tank/dsmil

# List snapshots
zfs list -t snapshot | grep dsmil

# Test snapshot
sudo zfs snapshot tank/dsmil@test
sudo zfs rollback tank/dsmil@test
```

---

## Troubleshooting

### Service Can't Find Files

**Error:** `FileNotFoundError` or `No such file or directory`

**Solution:**
```bash
# Check systemd service paths
sudo systemctl cat dsmil-server.service | grep -E "Working|ExecStart"

# Verify files exist
ls /path/in/service/file

# Fix and reload
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

### Permission Errors

**Error:** `Permission denied`

**Solution:**
```bash
# Check ownership
ls -la /new/path

# Fix ownership
sudo chown -R $USER:$USER /new/path

# For ZFS
sudo zfs allow $USER mount,snapshot,send,receive tank/dsmil
```

### Symlinks Broken

**Error:** Symlink points to old location

**Solution:**
```bash
# Find broken symlinks
find ~/.ollama -xtype l
find ~/.local/share/dsmil -xtype l

# Remove broken symlinks
find ~/.ollama -xtype l -delete
find ~/.local/share/dsmil -xtype l -delete

# Recreate correct symlinks
ln -s /new/path/models ~/.ollama/models
ln -s /new/path/rag ~/.local/share/dsmil/rag_index
```

---

## Examples

### Example 1: Install to /opt

```bash
sudo mkdir -p /opt/dsmil
sudo chown $USER:$USER /opt/dsmil
cd /opt/dsmil
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .
./install-complete.sh
sudo sed -i "s|$HOME/LAT5150DRVMIL|/opt/dsmil|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

### Example 2: Install to External SSD

```bash
# Mount external SSD
sudo mkdir -p /mnt/external-ssd
sudo mount /dev/sdX1 /mnt/external-ssd
sudo chown $USER:$USER /mnt/external-ssd

# Install
mkdir -p /mnt/external-ssd/dsmil
cd /mnt/external-ssd/dsmil
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .
./install-complete.sh

# Update paths
sudo sed -i "s|$HOME/LAT5150DRVMIL|/mnt/external-ssd/dsmil|g" /etc/systemd/system/dsmil-server.service
sed -i 's|"install_dir": ".*"|"install_dir": "/mnt/external-ssd/dsmil"|g' ~/.config/dsmil/config.json

# Add to /etc/fstab for auto-mount
echo "/dev/sdX1  /mnt/external-ssd  ext4  defaults,noatime  0  2" | sudo tee -a /etc/fstab

# Restart
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

### Example 3: ZFS with Automatic Snapshots

```bash
# Create ZFS dataset
sudo zfs create tank/dsmil
sudo zfs set compression=lz4 tank/dsmil
sudo zfs set atime=off tank/dsmil
sudo chown $USER:$USER /tank/dsmil

# Install
cd /tank/dsmil
git clone https://github.com/SWORDIntel/LAT5150DRVMIL .
./install-complete.sh

# Setup automatic snapshots (using zfs-auto-snapshot)
sudo apt install zfs-auto-snapshot
sudo zfs set com.sun:auto-snapshot=true tank/dsmil

# This creates:
# - Hourly snapshots (keep 24)
# - Daily snapshots (keep 7)
# - Weekly snapshots (keep 4)
# - Monthly snapshots (keep 12)

# Update paths
sudo sed -i "s|$HOME/LAT5150DRVMIL|/tank/dsmil|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

---

## Best Practices

### ✅ DO

- Use ZFS for production deployments (snapshots, integrity)
- Put models on large, slower drive (HDD) if space constrained
- Keep source code on fast drive (SSD/NVMe)
- Take snapshot before updates
- Test on clone before promoting to production
- Use compression (lz4 for speed, zstd for space)
- Set appropriate recordsize (1M for models, 16k for RAG)

### ❌ DON'T

- Install to network drive (NFS/CIFS) - too slow
- Use FAT32/exFAT - no permissions/symlinks
- Forget to update systemd paths
- Skip verification after transplant
- Delete old installation immediately (wait a week)
- Use RAID0 without backups

---

## Quick Reference

### Installation Locations

| Location | Pros | Cons | Best For |
|----------|------|------|----------|
| `/home/user/` | Easy, no sudo | Limited space | Development |
| `/opt/dsmil/` | Standard, multi-user | Needs sudo | Production |
| `/data/dsmil/` | Dedicated drive | Setup required | Large deployments |
| `tank/dsmil/` | ZFS snapshots | ZFS setup | Mission critical |
| `/mnt/ssd/` | Fast I/O | External mount | Performance |

### Commands

```bash
# Transplant
rsync -avh ~/LAT5150DRVMIL/ /new/path/
sudo sed -i "s|$HOME/LAT5150DRVMIL|/new/path|g" /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload && sudo systemctl restart dsmil-server

# ZFS snapshot
sudo zfs snapshot tank/dsmil@backup-$(date +%Y%m%d)

# ZFS rollback
sudo zfs rollback tank/dsmil@backup-20251030

# Verify
systemctl status dsmil-server
curl http://localhost:9876/status
```

---

## Support

**For questions:**
- See COMPLETE_INSTALLATION.md for general installation
- See INSTALL_IN_PLACE.md for existing system installation
- See SECURITY_CONFIG.md for security settings
- GitHub: https://github.com/SWORDIntel/LAT5150DRVMIL/issues

---

**Ready to transplant?** Follow the method that matches your setup! 🚀
