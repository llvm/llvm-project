#!/bin/bash
# Cleanup and organize AI-related files in /home/john

echo "Organizing AI project files..."

# Create archive directory for old session files
mkdir -p ~/LAT5150DRVMIL/00-documentation/session-archives

# Move session summaries
mv ~/SESSION_COMPLETE.txt ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null
mv ~/SESSION_FINAL_SUMMARY.txt ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null
mv ~/INSTALLATION_COMPLETE.txt ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null
mv ~/FINAL_DEPLOYMENT_STATUS.txt ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null
mv ~/TRANSPLANT_SESSION_COMPLETE.md ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null

# Move ZFS-related files to transplant docs
mv ~/CURRENT_SITUATION.txt ~/LAT5150DRVMIL/zfs-transplant-docs/ 2>/dev/null
mv ~/FIX_ZFSBOOTMENU.sh ~/LAT5150DRVMIL/zfs-transplant-docs/ 2>/dev/null
mv ~/REBOOT_NOW.txt ~/LAT5150DRVMIL/zfs-transplant-docs/ 2>/dev/null

# Move other project docs
mv ~/NEXT_STEPS.txt ~/LAT5150DRVMIL/00-documentation/ 2>/dev/null

# Move logs
mkdir -p ~/LAT5150DRVMIL/logs/kernel-builds
mv ~/rebuild-log.txt ~/LAT5150DRVMIL/logs/ 2>/dev/null
mv ~/ultimate-*.log ~/LAT5150DRVMIL/logs/kernel-builds/ 2>/dev/null
mv ~/install-now.log ~/LAT5150DRVMIL/logs/ 2>/dev/null
mv ~/quick-install.log ~/LAT5150DRVMIL/logs/ 2>/dev/null

# Move old scripts to archive
mkdir -p ~/LAT5150DRVMIL/99-archive/old-scripts
mv ~/bash_harvest.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/display-banner.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/enable-huge-pages.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/enable_dsmil_boot.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/harvest.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/install-dsmil-kernel.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/install_ollama_intel.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/launch-opus-interface.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/launch_64gram_pcore.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/post-reboot-check.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/quick-start-interface.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/show-complete-status.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/start-local-opus.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/start-opus-server.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/test-local-opus.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/verify-system.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/verify_system.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null
mv ~/START_SERVER.sh ~/LAT5150DRVMIL/99-archive/old-scripts/ 2>/dev/null

# Move old security reports to archives
mv ~/FINAL_SECURITY_REPORT.md ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null
mv ~/PERSISTENCE_AUDIT.md ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null
mv ~/SECURITY_FINDINGS.txt ~/LAT5150DRVMIL/00-documentation/session-archives/ 2>/dev/null

# Move old Opus files
mkdir -p ~/LAT5150DRVMIL/99-archive/opus-transfer
mv ~/COPY_THIS_TO_OPUS.txt ~/LAT5150DRVMIL/99-archive/opus-transfer/ 2>/dev/null
mv ~/OPUS_DIRECT_PASTE.txt ~/LAT5150DRVMIL/99-archive/opus-transfer/ 2>/dev/null
mv ~/URGENT_OPUS_TRANSFER.sh ~/LAT5150DRVMIL/99-archive/opus-transfer/ 2>/dev/null

# Move test files
mv ~/test-document.txt ~/LAT5150DRVMIL/99-archive/ 2>/dev/null

echo "Cleanup complete!"
echo ""
echo "Organized:"
echo "  Session docs → LAT5150DRVMIL/00-documentation/session-archives/"
echo "  ZFS docs → LAT5150DRVMIL/zfs-transplant-docs/"
echo "  Logs → LAT5150DRVMIL/logs/"
echo "  Old scripts → LAT5150DRVMIL/99-archive/old-scripts/"
echo "  Security reports → LAT5150DRVMIL/00-documentation/session-archives/"
echo ""
echo "Files remaining in ~/ (if any) are non-AI related"
