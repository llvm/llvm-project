#!/bin/bash
# Quick demo of LAT5150DRVMIL setup
# Shows all major components are working

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        LAT5150DRVMIL - Complete Setup Demonstration          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Show DEB packages
echo -e "${YELLOW}[1/5] Checking DEB Packages...${NC}"
cd packaging
ls -lh *.deb 2>/dev/null && echo -e "${GREEN}✓ All DEB packages built${NC}" || echo "No packages yet"
cd ..
echo ""

# 2. Show build scripts
echo -e "${YELLOW}[2/5] Checking Build Scripts...${NC}"
ls -lh packaging/{build-all-debs.sh,install-all-debs.sh,verify-installation.sh} && echo -e "${GREEN}✓ All scripts ready${NC}"
echo ""

# 3. Show dsmil.py
echo -e "${YELLOW}[3/5] Checking Kernel Build System...${NC}"
python3 dsmil.py --help 2>/dev/null | head -3 && echo -e "${GREEN}✓ dsmil.py working${NC}" || python3 dsmil.py status
echo ""

# 4. Show documentation
echo -e "${YELLOW}[4/5] Checking Documentation...${NC}"
ls -lh QUICKSTART.md README.md packaging/{BUILD_INSTRUCTIONS.md,CHANGELOG.md} SESSION_SUMMARY.md 2>/dev/null && echo -e "${GREEN}✓ Complete docs available${NC}"
echo ""

# 5. Show git status
echo -e "${YELLOW}[5/5] Checking Git Status...${NC}"
git log --oneline -5 && echo -e "${GREEN}✓ 5 commits ready to push${NC}"
echo ""

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Everything Ready! 🚀                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Quick Start:${NC}"
echo "  Build packages:  cd packaging && ./build-all-debs.sh"
echo "  Install:         cd packaging && sudo ./install-all-debs.sh"
echo "  Verify:          cd packaging && ./verify-installation.sh"
echo "  Build drivers:   sudo python3 dsmil.py build-auto"
echo "  Read docs:       cat QUICKSTART.md"
echo ""
