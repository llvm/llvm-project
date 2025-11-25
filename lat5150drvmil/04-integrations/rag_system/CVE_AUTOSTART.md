# Automatic CVE Scraper - Timer-Based Execution

## Overview

The CVE scraper runs **fully automatically** using systemd timers. No manual intervention needed.

**Execution Schedule:**
- ✅ **On boot:** 1 minute after system starts
- ✅ **Every 5 minutes:** Continuous monitoring
- ✅ **Daily at 3 AM:** Full resync and cleanup
- ✅ **Persistent:** Catches up if system was off

---

## Quick Install

### One-Time Setup

```bash
# 1. Install dependencies
pip install telethon python-dotenv

# 2. Authenticate with Telegram (one-time only)
python3 rag_system/telegram_cve_scraper.py --oneshot

# Enter phone number and verification code when prompted
# This creates telegram_cve_session.session (auto-login forever)

# 3. Install systemd service
sudo ./rag_system/install_cve_service.sh
```

**That's it!** The scraper now runs automatically forever.

---

## How It Works

### Systemd Timer Architecture

```
┌─────────────────────────────────────┐
│   System Boot / Every 5 min         │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   cve-scraper.timer                 │
│   Triggers → cve-scraper.service    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   telegram_cve_scraper.py           │
│   --oneshot mode                    │
│   1. Check @cveNotify               │
│   2. Save new CVEs                  │
│   3. Update RAG if batch full       │
│   4. Exit                           │
└──────────────┬──────────────────────┘
               ↓
        Wait 5 minutes → Repeat
```

### One-Shot Mode

The scraper runs in **one-shot mode** (not continuous):
- Connects to Telegram
- Scrapes latest messages (last 100)
- Saves new CVEs
- Updates RAG if needed
- Exits cleanly

**Why one-shot?**
- More reliable (fresh connection each time)
- Lower resource usage
- Better error recovery
- Clean logs per run

---

## Installation Steps Detailed

### Step 1: Install Dependencies

```bash
pip install telethon python-dotenv
```

### Step 2: First-Time Authentication

```bash
# Run once to create session file
python3 rag_system/telegram_cve_scraper.py --oneshot
```

**You'll be prompted for:**
1. Phone number (with country code): `+12345678900`
2. Verification code (sent to Telegram app)
3. 2FA password (if you have it enabled)

**This creates:** `telegram_cve_session.session`
- Persistent login (never expires)
- Auto-authenticates on every run
- No more password prompts!

### Step 3: Install Systemd Service

```bash
sudo ./rag_system/install_cve_service.sh
```

**This does:**
- Copies service files to `/etc/systemd/system/`
- Enables automatic startup
- Starts the timer
- Shows next execution time

---

## Management Commands

### Check Status

```bash
# Timer status
sudo systemctl status cve-scraper.timer

# Service status
sudo systemctl status cve-scraper.service

# Next scheduled run
systemctl list-timers cve-scraper.timer
```

### View Logs

```bash
# Live logs (follow mode)
sudo journalctl -u cve-scraper -f

# Last 100 lines
sudo journalctl -u cve-scraper -n 100

# Today's logs
sudo journalctl -u cve-scraper --since today

# Specific time range
sudo journalctl -u cve-scraper --since "2025-11-08 00:00" --until "2025-11-08 23:59"
```

### Manual Control

```bash
# Trigger immediate run (don't wait for timer)
sudo systemctl start cve-scraper.service

# Stop timer (disable automatic runs)
sudo systemctl stop cve-scraper.timer

# Restart timer
sudo systemctl restart cve-scraper.timer

# Disable on boot
sudo systemctl disable cve-scraper.timer

# Re-enable on boot
sudo systemctl enable cve-scraper.timer
```

### Uninstall

```bash
sudo ./rag_system/uninstall_cve_service.sh
```

---

## Configuration

### Change Run Frequency

Edit `/etc/systemd/system/cve-scraper.timer`:

```ini
[Timer]
# Run every 1 minute (more frequent)
OnUnitActiveSec=1min

# Or every 30 minutes (less frequent)
OnUnitActiveSec=30min

# Or every hour
OnUnitActiveSec=1h
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart cve-scraper.timer
```

### Change Schedule Time

Edit the daily sync time:

```ini
[Timer]
# Run daily at 6 PM instead of 3 AM
OnCalendar=daily 18:00
```

### Adjust Batch Size

Edit `.env.telegram`:

```bash
# Update RAG after every 5 CVEs instead of 10
UPDATE_BATCH_SIZE=5

# Or update after every single CVE
UPDATE_BATCH_SIZE=1

# Or disable auto-updates entirely
AUTO_UPDATE_EMBEDDINGS=false
```

---

## Monitoring & Statistics

### Real-Time Stats

```bash
# Check CVE statistics
python3 rag_system/telegram_cve_scraper.py --stats
```

**Output:**
```
Total CVEs: 247
Last Update: 2025-11-08T12:34:56
Pending RAG Update: 3

By Severity:
  Critical  :   42
  High      :   89
  Medium    :   76
  Low       :   28
```

### Log Analysis

```bash
# Count CVEs scraped today
sudo journalctl -u cve-scraper --since today | grep "Found new CVE" | wc -l

# Show all critical CVEs found
sudo journalctl -u cve-scraper | grep "Critical"

# Check RAG update frequency
sudo journalctl -u cve-scraper | grep "RAG embeddings updated"
```

---

## Troubleshooting

### Issue: Service won't start

```bash
# Check service status
sudo systemctl status cve-scraper.service

# Check logs
sudo journalctl -u cve-scraper -n 50

# Common causes:
# 1. Session file missing → Run manual auth first
# 2. Dependencies missing → pip install telethon python-dotenv
# 3. Permissions → Check .env.telegram is readable
```

### Issue: Session expired

```bash
# Re-authenticate
python3 rag_system/telegram_cve_scraper.py --oneshot

# Then restart service
sudo systemctl restart cve-scraper.timer
```

### Issue: RAG not updating

```bash
# Check pending CVEs
python3 rag_system/telegram_cve_scraper.py --stats

# Force manual update
python3 rag_system/telegram_cve_scraper.py --update-rag

# Check batch size
cat .env.telegram | grep UPDATE_BATCH_SIZE
```

### Issue: Too many CVEs at once

```bash
# Telegram rate limiting
# Solution: Scraper automatically retries after wait period

# Check logs for FloodWaitError
sudo journalctl -u cve-scraper | grep FloodWait

# Reduce scraping frequency if needed
# Edit timer: OnUnitActiveSec=10min
```

---

## File Locations

```
System Files:
/etc/systemd/system/cve-scraper.service    # Service definition
/etc/systemd/system/cve-scraper.timer      # Timer schedule

Project Files:
/home/user/LAT5150DRVMIL/
├── .env.telegram                          # Credentials (gitignored)
├── telegram_cve_session.session           # Auth token (gitignored)
├── rag_system/
│   ├── telegram_cve_scraper.py            # Scraper script
│   ├── cve_scraper.log                    # Local logs
│   ├── cve_index.json                     # CVE database
│   ├── install_cve_service.sh             # Install script
│   └── uninstall_cve_service.sh           # Uninstall script
└── 00-documentation/CVE_Feed/
    └── CVE-*.md                           # Individual CVEs
```

---

## Systemd Timer Schedule Examples

### Default (Every 5 minutes)
```ini
OnBootSec=1min
OnUnitActiveSec=5min
```

### High Frequency (Every minute)
```ini
OnBootSec=30s
OnUnitActiveSec=1min
```

### Low Frequency (Every hour)
```ini
OnBootSec=5min
OnUnitActiveSec=1h
```

### Specific Times (8 AM, 12 PM, 6 PM)
```ini
OnCalendar=08:00
OnCalendar=12:00
OnCalendar=18:00
```

### Business Hours Only (9 AM - 5 PM, every hour)
```ini
OnCalendar=Mon..Fri 09..17:00
```

---

## Performance Impact

**Resource Usage:**
- **CPU:** <5% during 30-second scrape
- **Memory:** ~100MB
- **Disk I/O:** ~2KB per CVE
- **Network:** Minimal (only new messages)

**System Impact:**
- Negligible on LAT5150DRVMIL
- Runs in background with low priority
- Auto-throttles on rate limits

---

## Security Features

1. **Sandboxed Execution:**
   - `NoNewPrivileges=true`
   - `PrivateTmp=true`
   - `ProtectSystem=strict`

2. **Limited Permissions:**
   - Read-only home directory
   - Write only to RAG directories

3. **Automatic Restart:**
   - Crashes handled gracefully
   - 60-second restart delay

---

## Success Verification

After installation, verify everything works:

```bash
# 1. Check timer is active
sudo systemctl is-active cve-scraper.timer
# Should output: active

# 2. Check next run time
systemctl list-timers cve-scraper.timer
# Should show next trigger time

# 3. Trigger manual run
sudo systemctl start cve-scraper.service

# 4. Watch logs
sudo journalctl -u cve-scraper -f
# Should see "Scraping X messages from @cveNotify"

# 5. Check CVE files created
ls -la 00-documentation/CVE_Feed/
# Should see CVE-*.md files

# 6. Verify RAG can query them
python3 rag_system/transformer_query.py
> "Show recent CVEs"
```

---

## Unattended Operation

Once installed, the system is **fully autonomous:**

✅ Starts automatically on boot
✅ Runs every 5 minutes
✅ Self-recovers from errors
✅ Auto-updates RAG embeddings
✅ Logs all activity
✅ No manual intervention needed

**Perfect for:**
- Servers
- Always-on workstations
- Security monitoring systems
- Automated threat intelligence

---

**Status:** Fully automated, zero-maintenance CVE monitoring
**Installation:** 3 commands, 2 minutes
**Maintenance:** None required (check logs occasionally)
