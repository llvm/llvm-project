# Automated CVE Monitoring & RAG Integration

## Overview

Automatic CVE scraping from Telegram channel `@cveNotify` with real-time RAG system updates.

**Features:**
- ✅ Real-time CVE monitoring from Telegram
- ✅ Automatic parsing of CVE ID, CVSS score, severity
- ✅ Individual CVE markdown files for RAG ingestion
- ✅ Automatic RAG embedding updates (batch processing)
- ✅ Background service mode
- ✅ Persistent CVE index

---

## Quick Start

### 1. Setup (First Time)

```bash
# Install dependencies
pip install telethon python-dotenv

# Telegram credentials are already configured in .env.telegram
# (API ID: 37733572, Hash: 5fbbca6becf772efa224be5af735ce66)

# First run - authenticate with Telegram
python3 rag_system/telegram_cve_scraper.py --history 50

# You'll be prompted for:
#   - Phone number (with country code)
#   - Verification code (sent via Telegram)
#   - 2FA password (if enabled)
```

### 2. Run as Service

```bash
# Start background scraper
./rag_system/cve_service.sh start

# Check status
./rag_system/cve_service.sh status

# View live logs
./rag_system/cve_service.sh logs

# Stop service
./rag_system/cve_service.sh stop
```

---

## How It Works

### 1. CVE Detection

Monitors `@cveNotify` for messages containing:
- CVE IDs (e.g., CVE-2024-1234)
- CVSS scores
- Severity levels (Critical/High/Medium/Low)
- References and URLs

### 2. Data Storage

Each CVE is saved as a markdown file:

```
00-documentation/CVE_Feed/
├── CVE-2024-1234.md
├── CVE-2024-5678.md
└── CVE-2024-9999.md
```

**File format:**
```markdown
# CVE-2024-1234

**Severity:** Critical
**CVSS Score:** 9.8
**Discovered:** 2025-11-08T12:34:56

## Description
[Full CVE description from Telegram message]

## References
- https://nvd.nist.gov/vuln/detail/CVE-2024-1234
- https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-1234

## Metadata
- **Source:** Telegram @cveNotify
- **Added to RAG:** 2025-11-08T12:34:56
- **Category:** Security / CVE
```

### 3. RAG Integration

**Automatic Updates:**
- Batch size: 10 CVEs (configurable)
- Update interval: 5 minutes (configurable)
- Rebuilds document index + transformer embeddings

**Manual Update:**
```bash
./rag_system/cve_service.sh update
```

---

## Configuration

Edit `.env.telegram` to customize:

```bash
# Telegram credentials
TELEGRAM_API_ID=37733572
TELEGRAM_API_HASH=5fbbca6becf772efa224be5af735ce66

# Channel to monitor
CVE_CHANNEL=cveNotify

# Auto-update settings
AUTO_UPDATE_EMBEDDINGS=true
UPDATE_BATCH_SIZE=10          # Update RAG after N new CVEs
UPDATE_INTERVAL_SECONDS=300   # Check for updates every 5 min
```

---

## Usage Examples

### Scrape Historical CVEs

```bash
# Scrape last 100 messages
python3 rag_system/telegram_cve_scraper.py --history 100

# Scrape last 500 messages (first-time setup)
python3 rag_system/telegram_cve_scraper.py --history 500
```

### Real-Time Monitoring

```bash
# Start monitoring (foreground)
python3 rag_system/telegram_cve_scraper.py

# Start monitoring (background service)
./rag_system/cve_service.sh start
```

### Statistics

```bash
# Show CVE statistics
python3 rag_system/telegram_cve_scraper.py --stats

# Or via service
./rag_system/cve_service.sh status
```

**Output:**
```
======================================================================
CVE Scraper Statistics
======================================================================

Total CVEs: 247
Last Update: 2025-11-08T12:34:56
Pending RAG Update: 3

By Severity:
  Critical  :   42
  High      :   89
  Medium    :   76
  Low       :   28
  Unknown   :   12
```

---

## Query CVEs via RAG

Once CVEs are indexed, query them via the RAG system:

```bash
# Interactive query
python3 rag_system/transformer_query.py

# Example queries:
> "Show me critical CVEs from 2024"
> "What are recent authentication bypass vulnerabilities?"
> "CVEs affecting Linux kernel"
```

---

## Monitoring & Logs

### Live Logs

```bash
# Follow logs in real-time
tail -f rag_system/cve_scraper.log

# Or via service
./rag_system/cve_service.sh logs
```

### Log Format

```
2025-11-08 12:34:56 - INFO - CVE Scraper initialized
2025-11-08 12:35:01 - INFO - Scraping 100 messages from @cveNotify...
2025-11-08 12:35:05 - INFO - Found new CVE: CVE-2024-1234 (Critical)
2025-11-08 12:35:10 - INFO - Scraped 15 new CVEs from history
2025-11-08 12:35:15 - INFO - Updating RAG embeddings with 15 new CVEs...
2025-11-08 12:37:20 - INFO - ✓ RAG embeddings updated successfully
2025-11-08 12:38:00 - INFO - 🆕 New CVE detected: CVE-2024-5678 (High)
```

---

## Troubleshooting

### Issue: "Telegram credentials not found"

**Solution:**
Check `.env.telegram` file exists and contains:
```bash
TELEGRAM_API_ID=37733572
TELEGRAM_API_HASH=5fbbca6becf772efa224be5af735ce66
```

### Issue: "FloodWaitError: A wait of X seconds is required"

**Solution:**
Telegram rate limiting. The scraper will automatically retry after the wait period.

### Issue: "Could not find the input entity for @cveNotify"

**Solution:**
1. Make sure you've authenticated with Telegram first
2. Join the channel manually: https://t.me/cveNotify
3. Restart the scraper

### Issue: RAG embeddings not updating

**Solution:**
```bash
# Check pending CVEs
python3 rag_system/telegram_cve_scraper.py --stats

# Force update
./rag_system/cve_service.sh update
```

---

## Architecture

```
Telegram Channel (@cveNotify)
         ↓
   telegram_cve_scraper.py
         ↓
   ┌─────────────────────┐
   │  CVE Parser         │
   │  - Extract CVE ID   │
   │  - Parse CVSS       │
   │  - Extract severity │
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │  CVE Storage        │
   │  - Individual .md   │
   │  - cve_index.json   │
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │  RAG Update         │
   │  - Rebuild index    │
   │  - Regen embeddings │
   └─────────────────────┘
         ↓
   Queryable via transformer_query.py
```

---

## Security Considerations

1. **API Credentials:** Stored in `.env.telegram` (gitignored)
2. **Session File:** `telegram_cve_session.session` contains auth token (gitignored)
3. **Rate Limiting:** Automatic handling of Telegram flood waits
4. **CVE Validation:** All CVE IDs validated via regex

---

## Advanced Usage

### Custom CVE Filtering

Edit `telegram_cve_scraper.py` to add filters:

```python
# In CVEParser.parse_cve_message():
cvss_score = CVEParser.extract_cvss_score(message)

# Only save critical/high CVEs
if cvss_score and cvss_score < 7.0:
    return {}  # Skip medium/low severity
```

### Integration with Notifications

Add webhook notifications when critical CVEs are found:

```python
# In CVEScraper.monitor_new_cves():
if cve_data.get('severity') == 'Critical':
    # Send notification
    import requests
    requests.post('YOUR_WEBHOOK_URL', json=cve_data)
```

---

## Performance

- **Scraping speed:** ~100 messages/minute
- **Storage:** ~2KB per CVE markdown file
- **RAG update:** ~2-3 minutes for batch of 10 CVEs
- **Memory:** ~100MB for scraper + RAG update

---

## Files Created

```
.env.telegram                        # Telegram credentials (gitignored)
telegram_cve_session.session         # Auth session (gitignored)

rag_system/
├── telegram_cve_scraper.py          # Main scraper
├── cve_service.sh                   # Service manager
├── cve_scraper.log                  # Scraper logs
├── cve_index.json                   # CVE index
└── CVE_README.md                    # This file

00-documentation/CVE_Feed/
└── CVE-YYYY-NNNNN.md               # Individual CVE files
```

---

## Systemd Service (Optional)

For production deployment:

```bash
# Create systemd service
sudo nano /etc/systemd/system/cve-scraper.service
```

```ini
[Unit]
Description=LAT5150DRVMIL CVE Scraper
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/user/LAT5150DRVMIL
ExecStart=/usr/bin/python3 rag_system/telegram_cve_scraper.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable cve-scraper
sudo systemctl start cve-scraper
sudo systemctl status cve-scraper
```

---

**Status:** Ready to deploy
**Channel:** https://t.me/cveNotify
**Auto-update:** Enabled (batch size: 10)
