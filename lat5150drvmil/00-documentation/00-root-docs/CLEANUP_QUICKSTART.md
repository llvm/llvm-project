# Root Cleanup - Quick Start Guide

**JANITOR Agent - Claude Agent Framework v7.0**

## TL;DR - Execute Cleanup in 3 Steps

### Step 1: Create Backup (REQUIRED)
```bash
cd /home/john/LAT5150DRVMIL
tar -czf ~/LAT5150DRVMIL-root-backup-$(date +%Y%m%d-%H%M%S).tar.gz --exclude='.git' --exclude='99-archive' .
```

### Step 2: Dry Run (RECOMMENDED)
```bash
./safe-delete-root-artifacts.sh
# Review output - NO changes are made
```

### Step 3: Execute (WHEN READY)
```bash
DRY_RUN=0 ./safe-delete-root-artifacts.sh
# Follow prompts
```

## What Gets Cleaned

| Category | Count | Action |
|----------|-------|--------|
| Python scripts | 53 | Move to organized dirs |
| Shell scripts | 20 | Move to organized dirs |
| C source files | 15 | Move to 01-source/ |
| Compiled binaries | 7 | Delete (rebuild from source) |
| Old backups | 5 dirs | Delete |
| Old environments | 2 dirs (26MB) | Archive then delete |
| Mock/test data | 3 dirs | Delete |
| Config files | 10+ | Move to 02-deployment/config/ |
| Log files | 8+ | Move to 99-archive/old-logs/ |

**Total:** ~123 files and ~25 directories → Clean root with only 5-10 essential items

## After Cleanup - Root Will Contain

```
LAT5150DRVMIL/
├── .git/                    # Git repo
├── .github/                 # GitHub workflows
├── .gitignore               # NEW - prevents future clutter
├── README.md                # Project overview
├── 00-documentation/        # All docs
├── 01-source/               # All source code
├── 02-deployment/           # Deployment tools
├── 03-security/             # Security tools
├── 99-archive/              # Historical data
├── packaging/               # Packages
└── tpm2_compat/             # TPM2 layer
```

**No scripts, binaries, or artifacts in root!**

## Safety Features

- Dry-run mode by default
- Full backup before changes
- All deletions backed up to 99-archive/
- Detailed logging
- 100% reversible

## Common Questions

**Q: Is this safe?**
A: Yes. Dry-run first, full backups, everything is reversible.

**Q: What if something breaks?**
A: Restore from `99-archive/root-cleanup-backup-*/` or `~/LAT5150DRVMIL-root-backup-*.tar.gz`

**Q: Will this affect git?**
A: No. .gitignore prevents tracking build artifacts. Use `git rm -r --cached . && git add .` to update index.

**Q: How long does it take?**
A: 2-5 minutes to execute, instant to dry-run.

**Q: Can I undo this?**
A: Yes. All changes are backed up. Script creates `99-archive/root-cleanup-backup-TIMESTAMP/` with everything.

## Verification After Cleanup

```bash
# Should show clean root
ls -la /home/john/LAT5150DRVMIL/

# Should show organized scripts
ls -la /home/john/LAT5150DRVMIL/02-deployment/scripts/

# Should show organized source
ls -la /home/john/LAT5150DRVMIL/01-source/tests/

# Should show minimal untracked files
git status
```

## Need More Details?

See **ROOT_CLEANUP_PLAN.md** for complete documentation including:
- Full inventory of all 123 files
- Exact destination for each file
- LAT5150_DEV vs LAT5150_PROD comparison
- Troubleshooting guide
- Post-cleanup maintenance

## Files Created

1. `/home/john/LAT5150DRVMIL/safe-delete-root-artifacts.sh` - Cleanup script
2. `/home/john/LAT5150DRVMIL/.gitignore` - Prevents future clutter
3. `/home/john/LAT5150DRVMIL/ROOT_CLEANUP_PLAN.md` - Complete documentation
4. `/home/john/LAT5150DRVMIL/CLEANUP_QUICKSTART.md` - This file

## Execute Now

```bash
cd /home/john/LAT5150DRVMIL

# 1. Backup
tar -czf ~/LAT5150DRVMIL-root-backup-$(date +%Y%m%d-%H%M%S).tar.gz --exclude='.git' .

# 2. Dry run
./safe-delete-root-artifacts.sh

# 3. Execute (when ready)
DRY_RUN=0 ./safe-delete-root-artifacts.sh
```

**Ready to clean! Professional root directory in under 5 minutes.**
