# Telegram Security Feed Scrapers

Comprehensive Telegram scraping suite for automated security intelligence gathering and RAG database population.

## Overview

This suite provides three complementary tools:

1. **`telegram_cve_scraper.py`** - Original CVE-focused scraper (lightweight, CVE-only)
2. **`telegram_document_scraper.py`** - Enhanced multi-channel scraper with file downloads
3. **`vxunderground_archive_downloader.py`** - VX Underground paper archive downloader

## Features

### Enhanced Telegram Document Scraper

- ✅ **Multi-channel monitoring** - cveNotify, Pwn3rzs, and custom channels
- ✅ **File attachment downloads** - .md, .pdf, .txt, .doc, .docx, .json, .yaml, .sh, .py
- ✅ **CVE parsing** - Automatic CVE-ID, CVSS, and severity extraction
- ✅ **General document parsing** - Categorization (exploit, malware, research, tools, etc.)
- ✅ **Deduplication** - SHA256 hash-based duplicate detection
- ✅ **Automatic RAG updates** - Batch processing for efficient embedding generation
- ✅ **Real-time monitoring** - Live message and file capture
- ✅ **Historical scraping** - Backfill from channel history

### VX Underground Archive Downloader

- ✅ **APT reports** - APT1, APT28, APT29, Lazarus Group, Equation Group, etc.
- ✅ **Malware analysis** - Stuxnet, WannaCry, NotPetya, Emotet, TrickBot, Ryuk
- ✅ **Technique papers** - Code injection, anti-debugging, obfuscation, persistence
- ✅ **Git clone support** - Download entire VXUG-Papers repository
- ✅ **Automatic categorization** - APT, Malware Analysis, Techniques, DFIR, etc.
- ✅ **Deduplication** - SHA256-based
- ✅ **RAG integration** - Automatic embedding updates

---

## Quick Start

### 1. Install Dependencies

```bash
pip install telethon python-dotenv
```

### 2. Get Telegram API Credentials

1. Visit https://my.telegram.org/apps
2. Log in with your phone number
3. Create a new application
4. Copy your `api_id` and `api_hash`

### 3. Configure Environment

```bash
cd /home/user/LAT5150DRVMIL

# Copy template
cp .env.telegram.template .env.telegram

# Edit with your credentials
nano .env.telegram
```

Set your credentials:

```bash
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
SECURITY_CHANNELS=cveNotify,Pwn3rzs
```

### 4. First Run - Authentication

On first run, Telegram will send you a code via the app:

```bash
cd rag_system
python3 telegram_document_scraper.py
```

You'll be prompted:
```
Please enter your phone number (international format): +1234567890
Please enter the code you received: 12345
```

After authentication, a session file is created (`telegram_document_session`). Future runs won't require login.

---

## Usage Examples

### Enhanced Document Scraper

#### Monitor All Channels (Real-time + History)

```bash
python3 telegram_document_scraper.py
```

This will:
- Scrape last 200 messages from each channel
- Start real-time monitoring
- Download .md, .pdf, and other documentation files
- Auto-update RAG embeddings every 10 documents

#### One-Shot Mode (For Cron/Systemd)

```bash
python3 telegram_document_scraper.py --oneshot
```

Scrapes history, updates RAG, and exits. Perfect for scheduled tasks:

```cron
# Run every hour
0 * * * * cd /home/user/LAT5150DRVMIL/rag_system && python3 telegram_document_scraper.py --oneshot
```

#### History Only (No Monitoring)

```bash
python3 telegram_document_scraper.py --no-monitor --history 500
```

Scrapes last 500 messages and exits.

#### Statistics

```bash
python3 telegram_document_scraper.py --stats
```

Output:
```
================================================================================
Enhanced Security Scraper Statistics
================================================================================

Total CVEs: 1,234
Total Documents: 567
Total Files: 89
Last Update: 2025-11-09T14:30:00
Pending RAG Update: 3

CVEs by Severity:
  Critical  :   45
  High      :  234
  Medium    :  567
  Low       :  388

Documents by Category:
  exploit              :   89
  malware              :   67
  research             :  123
  vulnerability        :   45
  ...

Files by Extension:
  .pdf      :   45
  .md       :   23
  .txt      :   12
  ...
```

#### Force RAG Update

```bash
python3 telegram_document_scraper.py --update-rag
```

### VX Underground Archive Downloader

#### Download All Categories (Curated Lists)

```bash
python3 vxunderground_archive_downloader.py
```

Downloads from curated lists:
- APT reports (APT1, APT28, APT29, Lazarus, Equation Group, etc.)
- Malware analysis papers (Stuxnet, WannaCry, NotPetya, etc.)
- Technique papers (Code injection, obfuscation, persistence, etc.)

#### Download Entire Repository via Git

```bash
python3 vxunderground_archive_downloader.py --git
```

⚠️ **Warning:** This clones the entire VXUG-Papers repository, which is **large** (1GB+).

#### Limit Downloads Per Category

```bash
python3 vxunderground_archive_downloader.py --limit 10
```

Downloads only 10 papers per category (for testing).

#### Statistics

```bash
python3 vxunderground_archive_downloader.py --stats
```

Output:
```
================================================================================
VX Underground Archive Statistics
================================================================================

Total Papers: 156
Total Size: 2,345.67 MB
Last Update: 2025-11-09T14:45:00
Pending RAG Update: 0

Papers by Category:
  APT                  :   45
  Malware_Analysis     :   67
  Techniques           :   34
  DFIR                 :   10
```

---

## Directory Structure

After running the scrapers, files are organized as follows:

```
00-documentation/Security_Feed/
├── CVE-2024-XXXXX.md              # CVE documents
├── CVE-2025-YYYYY.md
├── Pwn3rzs/                       # Pwn3rzs channel documents
│   ├── Pwn3rzs_12345.md
│   ├── Pwn3rzs_12346.md
│   └── ...
├── Downloads/                     # File attachments
│   ├── cveNotify/
│   │   └── exploit_poc.py
│   ├── Pwn3rzs/
│   │   ├── research_paper.pdf
│   │   ├── technique_guide.md
│   │   └── tool_documentation.txt
│   └── ...
└── VX_Underground/                # VX Underground archive
    ├── APT/
    │   ├── APT1.pdf
    │   ├── APT28.pdf
    │   ├── vx_apt_0001_index.md
    │   └── ...
    ├── Malware_Analysis/
    │   ├── Stuxnet_Analysis.pdf
    │   ├── WannaCry_Analysis.pdf
    │   └── ...
    ├── Techniques/
    │   ├── Code_Injection_Techniques.pdf
    │   └── ...
    └── vxug_git_clone/            # Git clone (if used)
        └── ...
```

### Index Files

```
rag_system/
├── security_index.json             # Main security document index
│   ├── "cves": {...}               # CVE data
│   ├── "documents": {...}          # General documents
│   └── "files": {...}              # File attachments
│
└── vxunderground_index.json        # VX Underground index
    ├── "papers": {...}             # Paper metadata
    ├── "categories": {...}         # Category counts
    └── "download_progress": {...}  # Progress tracking
```

---

## Configuration

### Environment Variables (.env.telegram)

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_API_ID` | *Required* | Telegram API ID from my.telegram.org |
| `TELEGRAM_API_HASH` | *Required* | Telegram API hash from my.telegram.org |
| `SECURITY_CHANNELS` | `cveNotify,Pwn3rzs` | Comma-separated channel list (no @) |
| `AUTO_UPDATE_EMBEDDINGS` | `true` | Auto-update RAG embeddings |
| `UPDATE_BATCH_SIZE` | `10` | Update RAG after N new documents |
| `UPDATE_INTERVAL_SECONDS` | `300` | Periodic update interval (seconds) |

### Adding More Channels

Edit `.env.telegram`:

```bash
SECURITY_CHANNELS=cveNotify,Pwn3rzs,malware_traffic,exploit_db,zerodaytoday
```

Popular security channels:
- `cveNotify` - CVE announcements
- `Pwn3rzs` - Security research, exploits, tools
- `malware_traffic` - Malware traffic analysis
- `exploit_db` - Exploit database updates
- `zerodaytoday` - Zero-day news
- `threatpost` - Threat intelligence
- `bleepingcomputer` - Security news

### File Download Settings

In `telegram_document_scraper.py`:

```python
# Allowed file extensions
ALLOWED_EXTENSIONS = {'.md', '.pdf', '.txt', '.doc', '.docx', '.json', '.yaml', '.yml', '.sh', '.py'}

# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024
```

---

## RAG Integration

### Automatic Updates

When `AUTO_UPDATE_EMBEDDINGS=true`, the scraper automatically:

1. **Collects** new documents/files
2. **Batches** until `UPDATE_BATCH_SIZE` reached (default: 10)
3. **Runs** `document_processor.py` to index files
4. **Runs** `transformer_upgrade.py` to generate embeddings
5. **Clears** batch and repeats

### Manual RAG Update

Force update of all indexed documents:

```bash
# Telegram documents
python3 telegram_document_scraper.py --update-rag

# VX Underground papers
python3 vxunderground_archive_downloader.py --update-rag
```

### Querying RAG Database

After documents are indexed, query via your RAG system:

```python
# Example query
result = rag_query("What techniques does APT28 use for lateral movement?")
```

The RAG system will search across:
- CVE descriptions
- Pwn3rzs documentation
- VX Underground APT reports
- Malware analysis papers
- Technique documentation

---

## Running as Background Service

### Systemd Service (Continuous Monitoring)

Create `/etc/systemd/system/telegram-security-scraper.service`:

```ini
[Unit]
Description=Telegram Security Feed Scraper
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/user/LAT5150DRVMIL/rag_system
ExecStart=/usr/bin/python3 telegram_document_scraper.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable telegram-security-scraper
sudo systemctl start telegram-security-scraper
sudo systemctl status telegram-security-scraper
```

View logs:

```bash
sudo journalctl -u telegram-security-scraper -f
```

### Systemd Timer (Periodic One-Shot)

If you prefer periodic scraping instead of continuous monitoring:

Create `/etc/systemd/system/telegram-scraper.service`:

```ini
[Unit]
Description=Telegram Security Scraper (One-Shot)

[Service]
Type=oneshot
User=your_username
WorkingDirectory=/home/user/LAT5150DRVMIL/rag_system
ExecStart=/usr/bin/python3 telegram_document_scraper.py --oneshot
```

Create `/etc/systemd/system/telegram-scraper.timer`:

```ini
[Unit]
Description=Run Telegram Scraper Every Hour

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:

```bash
sudo systemctl enable telegram-scraper.timer
sudo systemctl start telegram-scraper.timer
```

---

## Troubleshooting

### "Telethon not installed"

```bash
pip install telethon python-dotenv
```

### "Telegram credentials not found"

1. Check `.env.telegram` exists in project root
2. Verify `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` are set
3. Don't include quotes around values

### "FloodWaitError: A wait of X seconds is required"

Telegram rate limiting. The scraper will automatically wait and retry.

To reduce rate limits:
- Lower `--history` value
- Add delays between channels
- Use `--oneshot` with longer intervals

### "File already exists" / "Duplicate file (hash match)"

This is normal - the scraper skips duplicates to avoid wasting space and processing time.

### Session File Issues

If you get authentication errors:

```bash
rm telegram_document_session  # Delete session
python3 telegram_document_scraper.py  # Re-authenticate
```

### VX Underground Downloads Failing (404 errors)

VX Underground occasionally reorganizes their repository. If downloads fail:

1. Try `--git` method to clone the entire repository
2. Check https://github.com/vxunderground/VXUG-Papers for current structure
3. Update URL patterns in `vxunderground_archive_downloader.py`

---

## Performance and Resource Usage

### Telegram Scraper

- **CPU:** Low (~5% during scraping, <1% while monitoring)
- **Memory:** ~100-200 MB
- **Network:** 1-10 MB/hour (depends on file downloads)
- **Disk:** Depends on files downloaded (typically 10-100 MB/day)

### VX Underground Downloader

- **CPU:** Low (~10% during downloads)
- **Memory:** ~100 MB
- **Network:** High during initial download (1-5 GB for git clone)
- **Disk:** 1-10 GB (full archive)

### RAG Embedding Updates

- **CPU:** High (50-100% for 1-5 minutes per batch)
- **Memory:** 2-8 GB (depends on model size)
- **Time:** ~30 seconds per document (embedding generation)

**Recommendation:** Run on system with:
- 8 GB+ RAM
- 20 GB+ free disk space
- Reasonable internet connection

---

## Security Considerations

### API Credentials

- **Never commit** `.env.telegram` to version control
- Store credentials securely
- Use environment-specific credentials for production

### Downloaded Content

- Files from Telegram channels may contain **malicious code**
- VX Underground papers discuss **malware techniques**
- **Do NOT execute** downloaded scripts without inspection
- Consider running scrapers in isolated environment (VM, container)

### File Validation

The scrapers include:
- File extension validation
- Size limits (50 MB for Telegram, 100 MB for VX)
- SHA256 deduplication

But they do **NOT** include:
- Virus scanning
- Content analysis
- Sandboxing

**Recommendation:** Add virus scanning:

```bash
# Install ClamAV
sudo apt-get install clamav

# Scan downloads
clamscan -r 00-documentation/Security_Feed/Downloads/
```

---

## Advanced Usage

### Custom Parser for Specific Channels

Extend `DocumentParser` to handle channel-specific formats:

```python
class CustomParser(DocumentParser):
    @staticmethod
    def parse_exploit_db_message(message: str) -> Dict:
        # Custom parsing for exploit-db channel
        exploit_id = re.search(r'EDB-ID: (\d+)', message)
        # ...
```

### Filter Downloads by Keyword

Modify `_download_file()` to filter:

```python
# Only download if message contains specific keywords
if not any(kw in message.text.lower() for kw in ['apt', 'exploit', 'cve']):
    return None
```

### Export to Other Formats

Export index to CSV, JSON, or database:

```python
import pandas as pd

# Load index
with open('rag_system/security_index.json') as f:
    index = json.load(f)

# Export CVEs to CSV
df = pd.DataFrame.from_dict(index['cves'], orient='index')
df.to_csv('cves_export.csv')
```

---

## Comparison: Original vs Enhanced Scraper

| Feature | `telegram_cve_scraper.py` | `telegram_document_scraper.py` |
|---------|---------------------------|--------------------------------|
| **CVE Parsing** | ✅ | ✅ |
| **Multi-channel** | ✅ | ✅ |
| **File Downloads** | ❌ | ✅ (.md, .pdf, .txt, etc.) |
| **General Documents** | ❌ | ✅ (categorized) |
| **Deduplication** | By CVE-ID | By SHA256 hash |
| **Pwn3rzs Support** | ❌ | ✅ |
| **Channel-specific dirs** | ❌ | ✅ |
| **Use Case** | CVE monitoring only | Full intelligence gathering |

**Recommendation:**
- Use original for **CVE-only** monitoring (lightweight)
- Use enhanced for **comprehensive** security intelligence (Pwn3rzs, documents, files)

---

## Monitoring and Logs

### Log Files

```bash
# Telegram scraper logs
tail -f rag_system/document_scraper.log

# VX Underground downloader logs
tail -f rag_system/vxunderground_downloader.log
```

### Log Rotation

Configure logrotate for long-running services:

```bash
# /etc/logrotate.d/telegram-scraper
/home/user/LAT5150DRVMIL/rag_system/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

## Next Steps

1. ✅ **Configure** `.env.telegram` with your API credentials
2. ✅ **Test** authentication: `python3 telegram_document_scraper.py --stats`
3. ✅ **Run initial scrape**: `python3 telegram_document_scraper.py --history 100 --no-monitor`
4. ✅ **Download VX archive**: `python3 vxunderground_archive_downloader.py --limit 5`
5. ✅ **Verify RAG integration**: Check embeddings are updated
6. 🚀 **Deploy** as systemd service for continuous monitoring

---

**Document Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Last Updated:** 2025-11-09
**Version:** 2.0
**Author:** AI Framework Team
