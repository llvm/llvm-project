# LAT5150 DRVMIL - Xen Hypervisor Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the LAT5150 DRVMIL Tactical Interface with Xen hypervisor VMs. The integration maintains strict security boundaries while enabling secure access from isolated virtual machines.

## Security Philosophy

**Principle:** Defense-in-Depth with VM Isolation

The tactical interface is designed to be localhost-only on the host system. For Xen VM access, we provide two secure methods:

1. **SSH Tunneling (Recommended):** Maximum security, localhost-only maintained
2. **Bridge Network Access:** Controlled network access with encryption and IP filtering

Both methods maintain TEMPEST compliance and operational security requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Dom0 (Xen Host)                                                 │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Tactical Interface (127.0.0.1:5001)                       │ │
│  │ - Self-coding system                                      │ │
│  │ - APT-grade security                                      │ │
│  │ - TEMPEST compliant                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Xen Bridge (xenbr0): 192.168.100.1                            │
│  ├─ SSH Server (optional)                                      │
│  └─ Nginx Proxy: 192.168.100.1:8443 (optional)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Isolated Network
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│ DomU (Xen VM - Tactical Client)                                 │
│                                                                  │
│  IP: 192.168.100.10                                             │
│                                                                  │
│  Method 1: SSH Tunnel                                           │
│  └─> ssh -L 5001:127.0.0.1:5001 root@192.168.100.1            │
│      Access: http://localhost:5001                              │
│                                                                  │
│  Method 2: Bridge HTTPS                                         │
│  └─> Direct HTTPS connection                                    │
│      Access: https://192.168.100.1:8443                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Xen Host (Dom0)

**Required:**
- Xen hypervisor 4.11+ installed and running
- Xen bridge network configured (xenbr0)
- SSH server (for tunneling method)
- OR Nginx (for bridge method)
- Root/sudo access

**Check Xen Installation:**
```bash
xl info
xl list
ip addr show xenbr0
```

### Xen VM (DomU)

**Required:**
- Linux-based VM (Debian/Ubuntu/CentOS/etc.)
- Network connectivity to host bridge
- SSH client (for tunneling)
- OR Web browser (for bridge access)

## Method 1: SSH Tunneling (Recommended)

### Security Benefits

- ✅ **Maximum Security:** Maintains localhost-only on host
- ✅ **Encrypted:** SSH encryption for all traffic
- ✅ **No Network Exposure:** Interface never exposed to network
- ✅ **TEMPEST Optimal:** Minimal electromagnetic signature
- ✅ **Simple Setup:** No additional services on host

### Host Setup (Dom0)

**1. Ensure SSH Server Running:**
```bash
# Check SSH status
systemctl status sshd

# If not running, start it
systemctl start sshd
systemctl enable sshd
```

**2. Configure SSH Access (Optional - Key-Based):**
```bash
# Create SSH key on VM (run from VM)
ssh-keygen -t ed25519 -C "tactical-client-vm"

# Copy public key to host
ssh-copy-id root@192.168.100.1
```

**3. No Other Configuration Needed!**
The tactical interface continues running on localhost:5001 as usual.

### VM Setup (DomU)

**1. Install SSH Client:**
```bash
# Debian/Ubuntu
apt-get install openssh-client

# RHEL/CentOS
yum install openssh-clients
```

**2. Copy Tunnel Script to VM:**
```bash
# From host, copy script to VM
scp /home/user/LAT5150DRVMIL/deployment/xen-vm-ssh-tunnel.sh root@192.168.100.10:/root/
```

**3. Run Tunnel Script:**
```bash
# On VM
chmod +x /root/xen-vm-ssh-tunnel.sh
./xen-vm-ssh-tunnel.sh 192.168.100.1
```

**4. Access Interface:**
```bash
# Open browser on VM
firefox http://localhost:5001
```

### Automatic Tunnel (SystemD Service)

Create a systemd service on the VM for automatic tunneling:

```bash
# Create service file on VM
cat > /etc/systemd/system/tactical-tunnel.service <<EOF
[Unit]
Description=SSH Tunnel to Tactical Interface
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/ssh -N -L 5001:127.0.0.1:5001 root@192.168.100.1 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable tactical-tunnel
systemctl start tactical-tunnel

# Check status
systemctl status tactical-tunnel
```

## Method 2: Bridge Network Access

### Security Benefits

- ✅ **HTTPS Encrypted:** TLS 1.3 encryption
- ✅ **IP Whitelisting:** Only VM network allowed
- ✅ **Firewall Protected:** iptables rules
- ✅ **Nginx Hardened:** Security headers, no server tokens
- ✅ **TEMPEST Compliant:** Bridge-only, not internet-exposed

### Security Considerations

- ⚠️ **Network Exposure:** Interface accessible over network (but restricted)
- ⚠️ **Certificate Trust:** Self-signed certificate requires manual trust
- ⚠️ **Additional Attack Surface:** Nginx as additional component

### Host Setup (Dom0)

**1. Run Configuration Script:**
```bash
cd /home/user/LAT5150DRVMIL/deployment
sudo ./configure_xen_bridge.sh install
```

This script will:
- Generate self-signed SSL certificates
- Configure Nginx reverse proxy
- Setup iptables firewall rules
- Copy tactical interface to web directory

**2. Verify Installation:**
```bash
sudo ./configure_xen_bridge.sh status
```

**3. Check Nginx:**
```bash
systemctl status nginx
ss -tlnp | grep 8443
```

**4. Get Bridge IP:**
```bash
ip addr show xenbr0 | grep "inet "
# Example output: 192.168.100.1/24
```

### VM Setup (DomU)

**1. Add Host to /etc/hosts (Optional but Recommended):**
```bash
# On VM
echo "192.168.100.1  tactical.xen.local" >> /etc/hosts
```

**2. Trust Self-Signed Certificate:**

**Firefox:**
1. Navigate to https://192.168.100.1:8443
2. Click "Advanced"
3. Click "Accept the Risk and Continue"

**Chromium:**
1. Navigate to https://192.168.100.1:8443
2. Click "Advanced"
3. Click "Proceed to 192.168.100.1 (unsafe)"

**3. Access Interface:**
```bash
firefox https://192.168.100.1:8443
# OR
firefox https://tactical.xen.local:8443
```

### Uninstall Bridge Access

```bash
cd /home/user/LAT5150DRVMIL/deployment
sudo ./configure_xen_bridge.sh remove
```

## Xen VM Configuration

### Creating a Tactical Client VM

**1. Copy Template:**
```bash
sudo cp /home/user/LAT5150DRVMIL/deployment/xen-templates/tactical-client.cfg \
        /etc/xen/tactical-client.cfg
```

**2. Customize Configuration:**

Edit `/etc/xen/tactical-client.cfg`:

```python
# Change VM name if needed
name = "tactical-client-01"

# Adjust resources
memory = 2048
vcpus = 2

# Update disk path
disk = [ 'phy:/dev/vg0/tactical-client-root,xvda,w' ]

# Update MAC address (must be unique)
vif = [ 'bridge=xenbr0,mac=00:16:3e:01:01:01' ]

# Update VNC password
vncpasswd = "your-secure-password"
```

**3. Create VM Disk:**

**LVM-based (Recommended):**
```bash
# Create logical volume (20GB)
lvcreate -L 20G -n tactical-client-root vg0

# Install OS to disk
# Option A: Use xen-create-image
xen-create-image --hostname=tactical-client \
    --size=20Gb --dist=bullseye \
    --gateway=192.168.100.1 \
    --netmask=255.255.255.0 \
    --ip=192.168.100.10

# Option B: Manual installation
# Boot from ISO and install to /dev/xvda
```

**File-based:**
```bash
# Create disk image (20GB)
dd if=/dev/zero of=/var/lib/xen/images/tactical-client.img bs=1M count=20480

# Format and mount
mkfs.ext4 /var/lib/xen/images/tactical-client.img
# ... install OS ...
```

**4. Start VM:**
```bash
# Create and start VM
xl create /etc/xen/tactical-client.cfg

# Connect to console
xl console tactical-client

# OR use VNC
vncviewer localhost:5901  # VNC display 1
```

**5. Initial VM Configuration:**

Inside the VM:

```bash
# Configure static IP
cat >> /etc/network/interfaces <<EOF
auto eth0
iface eth0 inet static
    address 192.168.100.10
    netmask 255.255.255.0
    gateway 192.168.100.1
EOF

# Restart networking
systemctl restart networking

# Add DNS
echo "nameserver 192.168.100.1" > /etc/resolv.conf

# Add host to hosts file
echo "192.168.100.1  tactical-host" >> /etc/hosts

# Test connectivity
ping -c 3 192.168.100.1
```

## Multiple VMs

### Scaling to Multiple Clients

**1. Clone VM Configuration:**
```bash
for i in {01..05}; do
    cp /etc/xen/tactical-client.cfg /etc/xen/tactical-client-${i}.cfg

    # Update name
    sed -i "s/tactical-client/tactical-client-${i}/" \
        /etc/xen/tactical-client-${i}.cfg

    # Update MAC (increment last octet)
    sed -i "s/mac=00:16:3e:01:01:01/mac=00:16:3e:01:01:$(printf '%02d' $i)/" \
        /etc/xen/tactical-client-${i}.cfg

    # Update VNC display
    sed -i "s/vncdisplay = 1/vncdisplay = ${i}/" \
        /etc/xen/tactical-client-${i}.cfg
done
```

**2. Update IP Whitelist:**

For bridge method, expand allowed network:

```bash
# Edit Nginx config
sudo nano /etc/nginx/sites-available/tactical-xen-bridge

# Change:
allow 192.168.100.0/24;

# Reload
sudo systemctl reload nginx
```

**3. Start All VMs:**
```bash
for i in {01..05}; do
    xl create /etc/xen/tactical-client-${i}.cfg
done

# List all VMs
xl list
```

## Operational Security (OPSEC)

### Best Practices

**Compartmentalization:**
- ✅ Use separate VMs for different classification levels
- ✅ Use separate VMs for different operations
- ✅ Never mix classified and unclassified in same VM

**Network Isolation:**
- ✅ Xen bridge is isolated from external network
- ✅ No internet access from tactical VMs (recommended)
- ✅ Host firewall blocks external access

**Access Control:**
- ✅ SSH key-based authentication only
- ✅ Strong VNC passwords
- ✅ Regular password rotation

**Monitoring:**
- ✅ Enable Xen logging
- ✅ Monitor SSH access logs
- ✅ Review Nginx access logs (if using bridge)
- ✅ Use tactical interface audit logs

**TEMPEST Compliance:**
- ✅ Use SSH tunneling for maximum EMF reduction
- ✅ If using bridge, enable all TEMPEST features in interface
- ✅ Use Level A mode for classified operations
- ✅ Ensure physical security of host and VMs

### VM Security Hardening

**Inside Each VM:**

```bash
# 1. Minimal installation (no unnecessary packages)
apt-get autoremove --purge

# 2. Disable unnecessary services
systemctl disable bluetooth
systemctl disable cups

# 3. Firewall (allow only SSH and tactical)
apt-get install ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow from 192.168.100.1 to any port 22
ufw enable

# 4. Automatic updates
apt-get install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades

# 5. No root SSH (use keys only)
sed -i 's/PermitRootLogin yes/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
systemctl restart sshd
```

## Troubleshooting

### SSH Tunnel Issues

**Problem:** "Connection refused"
```bash
# Check SSH server on host
ssh root@192.168.100.1

# Check tactical interface running
curl http://127.0.0.1:5001/api/health
```

**Problem:** "Port already in use"
```bash
# Kill existing tunnel
pkill -f "ssh.*192.168.100.1.*5001"

# Try again
./xen-vm-ssh-tunnel.sh 192.168.100.1
```

**Problem:** "Permission denied (publickey)"
```bash
# Setup SSH key authentication
ssh-keygen -t ed25519
ssh-copy-id root@192.168.100.1
```

### Bridge Access Issues

**Problem:** "Connection timed out"
```bash
# Check Nginx running
systemctl status nginx

# Check listening port
ss -tlnp | grep 8443

# Check firewall
iptables -L INPUT -n | grep 8443
```

**Problem:** "403 Forbidden"
```bash
# Check IP whitelist in Nginx config
sudo nano /etc/nginx/sites-available/tactical-xen-bridge

# Verify VM IP is in allowed range
allow 192.168.100.0/24;

# Reload Nginx
sudo systemctl reload nginx
```

**Problem:** "SSL certificate error"
```bash
# This is expected with self-signed certs
# Click "Advanced" and "Accept Risk" in browser

# OR regenerate certificate
cd /etc/nginx/ssl
sudo rm tactical-xen.*
sudo ./configure_xen_bridge.sh install
```

### VM Network Issues

**Problem:** "Cannot reach host"
```bash
# Check VM networking
ip addr show eth0
ip route show

# Check bridge on host
ip addr show xenbr0

# Ping test
ping 192.168.100.1
```

**Problem:** "VM not getting IP"
```bash
# Check Xen network config
xl network-list tactical-client

# Restart networking in VM
systemctl restart networking

# Check DHCP (if used)
dhclient eth0
```

### Xen VM Issues

**Problem:** "VM won't start"
```bash
# Check detailed error
xl create -vvv /etc/xen/tactical-client.cfg

# Check disk exists
ls -lh /dev/vg0/tactical-client-root
# OR
ls -lh /var/lib/xen/images/tactical-client.img

# Check kernel/bootloader
xl dmesg | tail -50
```

**Problem:** "Cannot connect to console"
```bash
# Check VM is running
xl list

# Try VNC instead
vncviewer localhost:5901

# Check VM logs
xl dmesg
cat /var/log/xen/tactical-client.log
```

## Performance Optimization

### VM Resource Allocation

**Memory:**
```python
# Minimum for browser-based access
memory = 1024

# Recommended for comfortable use
memory = 2048

# For heavy operations
memory = 4096
```

**CPUs:**
```python
# Single CPU sufficient for light use
vcpus = 1

# Recommended
vcpus = 2

# Pin to specific cores for isolation
cpus = "2-3"  # Use cores 2 and 3
```

**Disk I/O:**
```bash
# Use LVM with SSD for best performance
lvcreate -L 20G -n tactical-client-root vg0

# Or use file-based with preallocated space
dd if=/dev/zero of=tactical.img bs=1M count=20480
```

### Network Optimization

**SSH Tunnel:**
```bash
# Enable compression for slow links
ssh -C -L 5001:127.0.0.1:5001 root@192.168.100.1

# Increase keep-alive
ssh -o ServerAliveInterval=30 -L 5001:127.0.0.1:5001 root@192.168.100.1
```

**Bridge Access:**
```nginx
# Enable Nginx caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=tactical_cache:10m;

location /api/ {
    proxy_cache tactical_cache;
    proxy_cache_valid 200 1m;
    # ... rest of config ...
}
```

## Maintenance

### Regular Tasks

**Daily:**
- Check VM status: `xl list`
- Verify tactical interface running: `curl http://localhost:5001/api/health`
- Review access logs: `tail /var/log/nginx/tactical-xen-access.log`

**Weekly:**
- Update VMs: `apt-get update && apt-get upgrade` (in each VM)
- Rotate VNC passwords
- Review SSH access logs: `grep "Accepted" /var/log/auth.log`

**Monthly:**
- Update Xen hypervisor
- Backup VM disk images
- Review and update firewall rules
- Test disaster recovery procedures

### Backup and Recovery

**Backup VM:**
```bash
# Stop VM
xl destroy tactical-client

# Backup disk
dd if=/dev/vg0/tactical-client-root of=/backup/tactical-client.img bs=4M
# OR
lvcreate -L 20G -s -n tactical-client-backup /dev/vg0/tactical-client-root

# Start VM
xl create /etc/xen/tactical-client.cfg
```

**Restore VM:**
```bash
# Stop VM
xl destroy tactical-client

# Restore disk
dd if=/backup/tactical-client.img of=/dev/vg0/tactical-client-root bs=4M

# Start VM
xl create /etc/xen/tactical-client.cfg
```

## Security Certifications

### TEMPEST Compliance

**SSH Tunneling (Method 1):**
- ✅ Maintains localhost-only interface (maximum TEMPEST)
- ✅ No additional electromagnetic emissions
- ✅ Suitable for all classification levels

**Bridge Access (Method 2):**
- ✅ HTTPS encryption (minimal additional emissions)
- ✅ Restricted to bridge network (isolated from external)
- ⚠️ Additional Nginx service (minimal EMF increase)
- ✅ Suitable for Secret and below with proper controls

### Classification Levels

| Method | UNCLASS | CUI | SECRET | TOP SECRET |
|--------|---------|-----|--------|------------|
| **SSH Tunnel** | ✅ | ✅ | ✅ | ✅ (with proper hardware) |
| **Bridge HTTPS** | ✅ | ✅ | ✅ | ⚠️ (additional review) |

**Top Secret Requirements:**
- TEMPEST-certified hardware (host and VMs)
- Shielded facility
- SSH tunneling method (recommended)
- Regular emanations testing
- Classified network (if using bridge)

## Appendix

### Quick Reference

**Start Tactical Interface (Host):**
```bash
python /home/user/LAT5150DRVMIL/03-web-interface/secured_self_coding_api.py
```

**Start VM:**
```bash
xl create /etc/xen/tactical-client.cfg
```

**Connect via SSH Tunnel (VM):**
```bash
./xen-vm-ssh-tunnel.sh 192.168.100.1
firefox http://localhost:5001
```

**Connect via Bridge (VM):**
```bash
firefox https://192.168.100.1:8443
```

**Stop VM:**
```bash
xl shutdown tactical-client
# OR force
xl destroy tactical-client
```

### Common Commands

```bash
# List all VMs
xl list

# VM console
xl console tactical-client

# VM info
xl info

# VM CPU usage
xl top

# Network info
xl network-list tactical-client

# Logs
xl dmesg
cat /var/log/xen/xen.log
```

### Network Diagram

```
Host (Dom0): 192.168.100.1
├─ lo: 127.0.0.1 (tactical interface: :5001)
├─ xenbr0: 192.168.100.1 (bridge network)
└─ eth0: <external IP> (internet)

VMs (DomU):
├─ tactical-client-01: 192.168.100.10
├─ tactical-client-02: 192.168.100.11
├─ tactical-client-03: 192.168.100.12
└─ tactical-client-04: 192.168.100.13
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-15
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Xen Compatibility:** Xen 4.11+
