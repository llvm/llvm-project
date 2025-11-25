#!/bin/bash
#
# Uninstall CVE Scraper systemd service
# Run with: sudo ./uninstall_cve_service.sh
#

if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root"
    echo "Usage: sudo ./uninstall_cve_service.sh"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  Uninstalling CVE Scraper Service"
echo "════════════════════════════════════════════════════════════"
echo

# Stop and disable
echo "Stopping CVE scraper..."
systemctl stop cve-scraper.timer
systemctl stop cve-scraper.service
systemctl disable cve-scraper.timer

echo "✓ Service stopped and disabled"
echo

# Remove files
echo "Removing service files..."
rm -f /etc/systemd/system/cve-scraper.service
rm -f /etc/systemd/system/cve-scraper.timer

echo "✓ Service files removed"
echo

# Reload systemd
systemctl daemon-reload

echo "✓ Systemd reloaded"
echo
echo "════════════════════════════════════════════════════════════"
echo "  Uninstallation Complete!"
echo "════════════════════════════════════════════════════════════"
echo
