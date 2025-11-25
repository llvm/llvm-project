#!/bin/bash
#
# Install CVE Scraper as systemd service
# Run with: sudo ./install_cve_service.sh
#

set -e

if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root"
    echo "Usage: sudo ./install_cve_service.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_USER="${SUDO_USER:-$USER}"

echo "════════════════════════════════════════════════════════════"
echo "  Installing CVE Scraper Service"
echo "════════════════════════════════════════════════════════════"
echo
echo "User: $INSTALL_USER"
echo "Directory: $SCRIPT_DIR"
echo

# Copy service files
echo "Installing systemd service files..."
cp "$SCRIPT_DIR/systemd/cve-scraper.service" /etc/systemd/system/
cp "$SCRIPT_DIR/systemd/cve-scraper.timer" /etc/systemd/system/

# Update service file with actual user
sed -i "s/%i/$INSTALL_USER/g" /etc/systemd/system/cve-scraper.service

# Set permissions
chmod 644 /etc/systemd/system/cve-scraper.service
chmod 644 /etc/systemd/system/cve-scraper.timer

echo "✓ Service files installed"
echo

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload
echo "✓ Systemd reloaded"
echo

# Enable and start timer
echo "Enabling CVE scraper timer..."
systemctl enable cve-scraper.timer
systemctl start cve-scraper.timer

echo "✓ Timer enabled and started"
echo

# Show status
echo "════════════════════════════════════════════════════════════"
echo "  Installation Complete!"
echo "════════════════════════════════════════════════════════════"
echo
echo "Service installed and will run automatically:"
echo "  • On boot (1 minute delay)"
echo "  • Every 5 minutes"
echo "  • Daily at 3 AM (full resync)"
echo
echo "Commands:"
echo "  Status:   sudo systemctl status cve-scraper.timer"
echo "  Logs:     sudo journalctl -u cve-scraper -f"
echo "  Stop:     sudo systemctl stop cve-scraper.timer"
echo "  Restart:  sudo systemctl restart cve-scraper.timer"
echo
echo "Next trigger:"
systemctl list-timers cve-scraper.timer
echo
