#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# Fix UEFI Boot Order - Make ZFSBootMenu Boot First
#═══════════════════════════════════════════════════════════════════════════════

SUDO_PASS="1786"

echo "═══════════════════════════════════════════════════════════════"
echo "  FIXING UEFI BOOT ORDER"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Current boot order:"
echo "1786" | sudo -S efibootmgr | grep "BootOrder"
echo ""

echo "Current first boot (BootCurrent):"
echo "1786" | sudo -S efibootmgr | grep "BootCurrent"
echo ""

echo "Available ZFS boot options:"
echo "1786" | sudo -S efibootmgr | grep -E "Boot0006|Boot000B|ZFS"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  FIXING BOOT ORDER"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Current order: 0004,0006,000B,0001,0007,0000,0002,0005,0008,0009,0003
# Need: 0006,0004,000B,0001,0007,0000,0002,0005,0008,0009,0003
# (Put ZFSBootMenu-Xen first)

echo "Setting Boot0006 (ZFSBootMenu-Xen) as first boot option..."
echo "1786" | sudo -S efibootmgr --bootorder 0006,0004,000B,0001,0007,0000,0002,0005,0008,0009,0003

echo ""
echo "New boot order:"
echo "1786" | sudo -S efibootmgr | grep "BootOrder"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  BOOT ORDER FIXED!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Now when you reboot:"
echo "  1. System will boot from ZFSBootMenu (Boot0006)"
echo "  2. Enter password: 1/0523/600260"
echo "  3. Select: livecd-xen-ai"
echo "  4. System boots into ZFS with Xen + DSMIL support"
echo ""
echo "Ready to reboot? Run: sudo reboot"
echo ""
