# LAT5150DRVMIL Project Organization

**Last Updated:** 2025-10-31
**Version:** 8.3.2 with ZFS Transplant Ready
**Status:** Clean and organized

---

## Directory Structure

```
LAT5150DRVMIL/
├── 00-documentation/              # All documentation
│   ├── 00-indexes/               # Project indexes
│   ├── 01-planning/              # Planning docs
│   ├── 02-analysis/              # Technical analysis
│   ├── 03-ai-framework/          # AI framework docs
│   ├── 04-progress/              # Progress reports
│   ├── 05-reference/             # Reference materials
│   ├── 00-root-docs/             # Misc root-level docs
│   ├── session-archives/         # Session summaries & reports (9 files)
│   ├── scripts/                  # Documentation scripts
│   └── archive/                  # Historical docs
│
├── 01-source/                    # DSMIL framework source
│   ├── kernel/                   # Kernel module
│   ├── kernel-driver/            # Kernel driver
│   ├── userspace-tools/          # Userspace utilities
│   ├── debugging/                # Debug tools
│   ├── systemd/                  # Systemd integration
│   └── tests/                    # Test suites
│
├── 02-ai-engine/                 # AI inference engine
│   ├── dsmil_ai_engine.py       # Main AI engine
│   ├── smart_router.py          # Smart model routing
│   ├── code_specialist.py       # Code generation
│   ├── local_claude_code.py     # Local code editing
│   ├── web_search.py            # Web search integration
│   ├── unified_orchestrator.py  # Multi-backend orchestration
│   └── sub_agents/              # Specialized sub-agents
│
├── 03-web-interface/            # Web UI and server
│   ├── clean_ui_v3.html         # Modern ChatGPT-style UI
│   ├── dsmil_unified_server.py  # Backend server (localhost-only)
│   ├── military_terminal_v2.html # Alternative terminal UI
│   └── RAG documentation
│
├── 03-security/                 # Security documentation
│   ├── audit/                   # Security audits
│   └── procedures/              # Safety procedures
│
├── 04-integrations/             # External integrations
│   ├── rag_manager.py           # RAG knowledge base
│   ├── web_scraper.py           # Intelligent web crawler
│   └── crawl4ai_wrapper.py      # Industrial crawler
│
├── 05-deployment/               # Deployment configs
│   ├── systemd/                 # Systemd service files
│   ├── npu-covert-edition.env   # Covert Edition env
│   └── verify_system.sh         # System verification
│
├── 99-archive/                  # Archived content
│   ├── old-scripts/             # Historical scripts (18 files)
│   ├── opus-transfer/           # Old Opus transfer files
│   ├── 02-deployment-backup/    # Old deployment dirs
│   ├── deployment-backup/       # Old deployment backup
│   ├── docs-backup/             # Old docs
│   └── test-document.txt        # Test files
│
├── logs/                        # Application logs
│   ├── kernel-builds/           # Kernel build logs
│   ├── rebuild-log.txt          # ZFS rebuild log
│   └── install logs             # Installation logs
│
├── packaging/                   # Debian packages
│   ├── dsmil-complete_8.3.2-1.deb (meta-package)
│   ├── dsmil-platform_8.3.1-1.deb (AI platform)
│   ├── dell-milspec-tools_*.deb
│   └── tpm2-accel-examples_*.deb
│
├── zfs-transplant-docs/         # ZFS transplant documentation (22 files)
│   ├── README.md                # Transplant docs index
│   ├── FINAL_REBOOT_CHECKLIST.txt # Complete pre-reboot status
│   ├── HANDOVER_TO_NEXT_AI.md  # Session handover
│   ├── SECURITY_FLAGS_STATUS.md # APT/Vault7 flags
│   ├── Installation scripts     # 6 automated installers
│   ├── Build scripts            # 3 kernel build scripts
│   └── Utility scripts          # 3 helper scripts
│
├── build/                       # Build artifacts
├── tpm2_compat/                 # TPM compatibility layer
│
├── install-complete.sh          # Complete installer (DSMIL + AI)
├── install.sh                   # Basic installer (AI only)
├── uninstall.sh                 # Uninstaller
├── cleanup.sh                   # Codebase cleanup
├── CLEANUP_HOME.sh              # Home directory cleanup
│
├── README.md                    # Main documentation
├── INSTALL.md                   # Basic install guide
├── COMPLETE_INSTALLATION.md     # Comprehensive guide (300+ lines)
├── INSTALL_IN_PLACE.md          # In-place install (400+ lines)
├── INSTALL_TO_DRIVE.md          # Custom drive install (500+ lines)
├── SECURITY_CONFIG.md           # Security configuration (350+ lines)
├── STRUCTURE.md                 # Directory layout
├── CLEANUP_REPORT.md            # Cleanup actions
├── INSTALLATION_SUMMARY.txt     # Quick reference
│
├── 00-ZFS-TRANSPLANT-STATUS.md  # ZFS transplant status
├── AI_FRAMEWORK_ZFS_TRANSPLANT.md # ZFS transplant guide
├── TRANSPLANT_TO_ZFS.sh         # ZFS transplant script
├── INSTALL_NOW.sh               # Quick install script
├── MANUAL_INSTALL_COMMANDS.txt  # Manual commands
│
└── COMPLETE_SESSION_CONTEXT_2025-10-31.md # Full session context
```

---

## File Categories

### Documentation (50+ files)
- Installation guides: 4 comprehensive guides
- Security documentation: 3 files
- Session archives: 9 session summaries
- Technical docs: 40+ in 00-documentation/
- ZFS transplant: 22 files in zfs-transplant-docs/

### Source Code
- DSMIL framework: 01-source/ (kernel modules, drivers)
- AI engine: 02-ai-engine/ (Python AI inference)
- Web interface: 03-web-interface/ (HTML/JS/Python server)
- Integrations: 04-integrations/ (RAG, web scraping)

### Scripts (40+ files)
- Installation: 4 main installers
- ZFS transplant: 9 scripts
- Old/archived: 18 scripts in 99-archive/
- Utilities: Various helper scripts

### Packages
- 4 .deb packages (2.5MB total)
- Ready for distribution

### Logs
- Kernel builds: 5 log files
- Installation: 4 log files
- System logs: Various

---

## What Got Cleaned Up

**Moved from ~/ to LAT5150DRVMIL/:**

**Session Documents (9 files):**
- SESSION_COMPLETE.txt
- SESSION_FINAL_SUMMARY.txt
- INSTALLATION_COMPLETE.txt
- FINAL_DEPLOYMENT_STATUS.txt
- TRANSPLANT_SESSION_COMPLETE.md
- FINAL_SECURITY_REPORT.md
- PERSISTENCE_AUDIT.md
- SECURITY_FINDINGS.txt
- NEXT_STEPS.txt

**ZFS Transplant (3 files):**
- CURRENT_SITUATION.txt
- FIX_ZFSBOOTMENU.sh
- REBOOT_NOW.txt

**Old Scripts (18 files):**
- Various bash scripts from early development
- Opus transfer scripts
- Test scripts
- System verification scripts

**Build Logs (5 files):**
- ultimate-build.log
- ultimate-build-actual.log
- ultimate-build-clean.log
- ultimate-build-FINAL.log
- rebuild-log.txt

**Total Organized:** 35+ files

---

## Clean Home Directory

**Remaining in ~/:**
- Personal files (Documents/, Desktop/, etc.)
- LAT5150DRVMIL/ (main project - organized)
- Other projects (livecd-gen, SpyGram, etc.)
- System config files (.bashrc, .config/, etc.)

**All AI project files now in:**
`/home/john/LAT5150DRVMIL/`

---

## Quick Access

**Main README:**
```bash
cat ~/LAT5150DRVMIL/README.md
```

**Reboot Checklist:**
```bash
cat ~/FINAL_REBOOT_CHECKLIST.txt
```

**Session Context:**
```bash
cat ~/LAT5150DRVMIL/COMPLETE_SESSION_CONTEXT_2025-10-31.md
```

**ZFS Transplant Docs:**
```bash
ls ~/LAT5150DRVMIL/zfs-transplant-docs/
```

---

## Git Status

**Repository:** https://github.com/SWORDIntel/LAT5150DRVMIL
**Latest Commit:** d663a58
**Total Commits This Session:** 8
**Files in Repo:** Organized and clean
**Ready for:** Production deployment

---

**Project is now professionally organized and ready for ZFS reboot!** 🚀
