# LAT5150 DRVMIL Tactical AI Sub-Engine - Quick Reference Guide

**Version:** 1.0.0 | **Classification:** TOP SECRET//SI//NOFORN

---

## Emergency Quick Start

```bash
# 1. Deploy everything
sudo /home/user/LAT5150DRVMIL/deployment/deploy-production.sh -y

# 2. Access tactical interface
firefox http://127.0.0.1:5001

# 3. Monitor system
sudo /home/user/LAT5150DRVMIL/deployment/monitor-system.sh
```

---

## Essential Commands

### Service Management
```bash
# Start service
sudo systemctl start lat5150-tactical.service

# Stop service
sudo systemctl stop lat5150-tactical.service

# Restart service
sudo systemctl restart lat5150-tactical.service

# Check status
sudo systemctl status lat5150-tactical.service

# View logs (live)
sudo journalctl -u lat5150-tactical.service -f

# View last 50 log lines
sudo journalctl -u lat5150-tactical.service -n 50
```

### Health Checks
```bash
# API health check
curl -s http://127.0.0.1:5001/health | jq

# Quick health check
curl -s http://127.0.0.1:5001/health | jq -r '.status'

# Response time test
time curl -s http://127.0.0.1:5001/health

# Check if service is listening
sudo netstat -tlnp | grep 5001
```

### Deployment Operations
```bash
# Full validation
sudo /home/user/LAT5150DRVMIL/deployment/validate-deployment.sh

# Monitor system in real-time
sudo /home/user/LAT5150DRVMIL/deployment/monitor-system.sh

# Deploy to production
sudo /home/user/LAT5150DRVMIL/deployment/deploy-production.sh

# Deploy with VMs
sudo /home/user/LAT5150DRVMIL/deployment/deploy-production.sh \
    --deploy-vms --vm-ips 192.168.100.10,192.168.100.11

# Deploy with DSMIL scan
sudo /home/user/LAT5150DRVMIL/deployment/deploy-production.sh \
    -y --scan-dsmil
```

### VM Operations
```bash
# Deploy shortcuts to single VM
cd /home/user/LAT5150DRVMIL/deployment
./deploy-vm-shortcuts.sh 192.168.100.10

# Deploy to multiple VMs
./deploy-vm-shortcuts.sh 192.168.100.10 192.168.100.11 192.168.100.12

# Check VM tunnels (from VM)
ps aux | grep "ssh.*5001:127.0.0.1:5001"

# Test VM access (from VM)
curl -s http://localhost:5001/health

# Restart tunnel (from VM)
pkill -f "ssh.*5001:127.0.0.1:5001"
/usr/local/bin/tactical-tunnel-autostart.sh
```

### DSMIL Hardware Operations
```bash
# Run enhanced reconnaissance
sudo /home/user/LAT5150DRVMIL/01-source/debugging/nsa_device_reconnaissance_enhanced.py

# View latest reconnaissance results
ls -t /home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced_*.json | head -1 | xargs cat | jq

# Check DSMIL device status
ls -l /dev/dsmil

# Check DSMIL logs
tail -f /home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced.log

# Count documented devices
ls -1 /home/user/LAT5150DRVMIL/00-documentation/devices/*.md | wc -l
```

---

## TEMPEST Display Modes

| Mode | Hotkey | EMF Reduction | Use Case |
|------|--------|---------------|----------|
| **Comfort (Level C)** | `Ctrl+1` | 45% | Default, extended operations |
| **Level A** | `Ctrl+2` | 80% | Top Secret operations |
| **Night Mode** | `Ctrl+3` | 55% | Low-light environments |
| **NVG Mode** | `Ctrl+4` | 70% | Night vision compatible |
| **High Contrast** | `Ctrl+5` | 35% | Accessibility |

**Switch Modes:**
1. Access tactical interface
2. Click "TACTICAL SETTINGS" (top-right)
3. Select desired mode
4. Adjust brightness/animations as needed

---

## Troubleshooting Quick Fixes

### Service Won't Start
```bash
# Check for errors
sudo journalctl -u lat5150-tactical.service -n 50

# Check port availability
sudo lsof -ti:5001

# Kill process on port 5001
sudo lsof -ti:5001 | xargs sudo kill -9

# Restart service
sudo systemctl restart lat5150-tactical.service
```

### API Not Responding
```bash
# Check if service is running
sudo systemctl status lat5150-tactical.service

# Check if port is listening
sudo netstat -tlnp | grep 5001

# Check for Python errors
sudo journalctl -u lat5150-tactical.service | grep -i error

# Restart API
sudo systemctl restart lat5150-tactical.service
```

### VM Can't Connect
```bash
# From VM: Check network connectivity
ping 192.168.100.1

# From VM: Check SSH connectivity
ssh root@192.168.100.1 "echo connected"

# From VM: Manually start tunnel
ssh -L 5001:127.0.0.1:5001 root@192.168.100.1 -N -f

# From VM: Check tunnel
lsof -i :5001
```

### DSMIL Device Not Found
```bash
# Check if device exists
ls -l /dev/dsmil

# Check kernel module
lsmod | grep dsmil

# Check device permissions
sudo chmod 666 /dev/dsmil

# Verify in reconnaissance
sudo python3 -c "import os; print(os.path.exists('/dev/dsmil'))"
```

---

## Network Ports & Bindings

| Port | Service | Binding | Purpose |
|------|---------|---------|---------|
| **5001** | Tactical API | 127.0.0.1 | Main interface (localhost only) |
| **22** | SSH | xenbr0 | VM tunnel access |
| **8443** | Nginx | xenbr0 | Optional bridge (if configured) |

**Verify Security:**
```bash
# Must show 127.0.0.1:5001 (NOT 0.0.0.0:5001)
sudo netstat -tlnp | grep 5001
```

---

## File Locations

### Core Files
```
/home/user/LAT5150DRVMIL/
├── 03-web-interface/
│   ├── secured_self_coding_api.py          # Main API server
│   └── tactical_self_coding_ui.html         # Tactical interface
├── 01-source/debugging/
│   └── nsa_device_reconnaissance_enhanced.py # DSMIL scanner
├── deployment/
│   ├── install-autostart.sh                 # Service installer
│   ├── deploy-vm-shortcuts.sh               # VM deployment
│   ├── validate-deployment.sh               # Validation tool
│   ├── monitor-system.sh                    # Monitoring tool
│   └── deploy-production.sh                 # Master deployment
├── DEPLOYMENT_GUIDE.md                      # Full deployment guide
├── PRODUCTION_READINESS_CHECKLIST.md        # Pre-deployment checklist
└── QUICK_REFERENCE.md                       # This document
```

### System Files
```
/etc/systemd/system/lat5150-tactical.service  # SystemD service
/dev/dsmil                                     # DSMIL device node
```

### VM Files (on guest VMs)
```
/usr/local/bin/tactical-tunnel-autostart.sh     # Tunnel script
/usr/share/applications/LAT5150-Tactical.desktop # Desktop launcher
~/.config/autostart/tactical-autostart.desktop   # Auto-start entry
~/Desktop/LAT5150-Tactical.desktop               # Desktop icon
```

---

## Performance Metrics

### Expected Performance
```
API Response Time:        < 100ms
UI Load Time:             < 2 seconds
Self-Coding Execution:    < 5 seconds
DSMIL Device Probe:       < 100ms per device
RAG Query:                < 1 second
VM Tunnel Latency:        < 10ms
```

### Resource Usage (Typical)
```
CPU Usage:                10-30% (idle to active)
Memory Usage:             2-6 GB
Disk Usage:               ~50 GB (with full embeddings)
Network (VM tunnels):     < 1 MB/s
```

---

## Security Checklist (Daily)

- [ ] Verify API bound to 127.0.0.1 only
- [ ] Check service logs for errors
- [ ] Verify VM tunnels encrypted (SSH)
- [ ] No external network exposure
- [ ] Firewall rules active

```bash
# Quick security check
sudo netstat -tlnp | grep 5001    # Must show 127.0.0.1
sudo iptables -L -n | grep 5001   # Firewall rules present
sudo journalctl -u lat5150-tactical.service -n 20 | grep -i error
```

---

## Backup & Recovery

### Quick Backup
```bash
# Create backup
sudo rsync -av --exclude='.git' \
    /home/user/LAT5150DRVMIL/ \
    /backup/LAT5150DRVMIL_$(date +%Y%m%d)/

# Verify backup
ls -lh /backup/LAT5150DRVMIL_*/
```

### Quick Recovery
```bash
# Stop service
sudo systemctl stop lat5150-tactical.service

# Restore from backup
sudo rsync -av /backup/LAT5150DRVMIL_YYYYMMDD/ \
    /home/user/LAT5150DRVMIL/

# Restart service
sudo systemctl start lat5150-tactical.service

# Verify health
curl -s http://127.0.0.1:5001/health
```

---

## Access URLs

### Primary Access (Host)
```
Tactical Interface:  http://127.0.0.1:5001
Health Check:        http://127.0.0.1:5001/health
API Documentation:   http://127.0.0.1:5001/docs
```

### VM Access (from Guest VMs)
```
Tactical Interface:  http://localhost:5001
Health Check:        http://localhost:5001/health
```

---

## Maintenance Schedule

### Daily
- [ ] Check service status
- [ ] Review logs for errors
- [ ] Monitor system performance

### Weekly
- [ ] Review full logs
- [ ] Check disk usage
- [ ] Verify backup integrity

### Monthly
- [ ] Run full validation
- [ ] Apply security updates
- [ ] Review and optimize performance

### Quarterly
- [ ] Run DSMIL reconnaissance
- [ ] Update device documentation
- [ ] Review and update procedures

---

## Support & Documentation

### Quick Help
```bash
# Service management help
systemctl --help

# Deployment help
/home/user/LAT5150DRVMIL/deployment/deploy-production.sh --help

# Monitoring tool help
/home/user/LAT5150DRVMIL/deployment/monitor-system.sh
# (Press 'h' for help in monitor)
```

### Full Documentation
- **Deployment Guide:** `/home/user/LAT5150DRVMIL/DEPLOYMENT_GUIDE.md`
- **Readiness Checklist:** `/home/user/LAT5150DRVMIL/PRODUCTION_READINESS_CHECKLIST.md`
- **Tactical UI Guide:** `/home/user/LAT5150DRVMIL/TACTICAL_INTERFACE_GUIDE.md`
- **TEMPEST Compliance:** `/home/user/LAT5150DRVMIL/TEMPEST_COMPLIANCE.md`
- **Xen Integration:** `/home/user/LAT5150DRVMIL/XEN_INTEGRATION_GUIDE.md`

---

## Contact & Escalation

**For Issues:**
1. Check this quick reference
2. Review relevant documentation
3. Check service logs
4. Run validation script
5. Escalate to system administrator

**Log Locations:**
- Service logs: `sudo journalctl -u lat5150-tactical.service`
- DSMIL logs: `/home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced.log`
- Deployment logs: `/tmp/lat5150_*.log`

---

**Print this document and keep it accessible for rapid reference during operations.**

**Classification:** TOP SECRET//SI//NOFORN
**Version:** 1.0.0
**Last Updated:** 2025-11-13
