# SPECTRA Integration Plan - Token-Efficient

## What SPECTRA Provides (Found in /home/john/SPECTRA)

✅ **Telegram Archiver** - Telethon-based, database-backed
✅ **Database Crawling** - SQLite/PostgreSQL with topic organization
✅ **Content Classification** - ML-based with auto-routing
✅ **Multi-Agent System** - Coordinated workflows
✅ **Web GUI** - Flask/SocketIO (Port 5000)

## Integration Into Our Opus Interface

### Phase 1: Add Telegram Button (10 min, ~5K tokens)

**Action**: Add Telegram channel archiver to sidebar
**Files**:
- Add button to `command_based_interface.html`
- Add endpoint to `opus_server_full.py`
- Link to existing SPECTRA `/home/john/SPECTRA`

**Command**:
```
telegram: @channelname
telegram crawl: @channel --depth 1000
```

### Phase 2: SPECTRA Database Access (5 min, ~3K tokens)

**Action**: Query SPECTRA SQLite DB from interface
**Files**:
- Read `/home/john/SPECTRA/*.db`
- Add `/spectra/query` endpoint
- Display in output window

**Command**:
```
spectra: search keyword
spectra: stats
```

### Phase 3: Launch SPECTRA GUI (2 min, ~2K tokens)

**Action**: Launch button that starts SPECTRA on port 5000
**Implementation**:
```python
subprocess.run(['python3', '/home/john/SPECTRA/spectra_gui_launcher.py'])
```

**Access**: http://localhost:5000 (separate from our port 9876)

## Minimal Token Integration (Total: ~10K tokens)

### Step 1: Add SPECTRA Endpoint (3K tokens)
```python
def spectra_operation(self):
    query = parse_qs(self.path.split('?')[1])
    cmd = query.get('cmd', [''])[0]

    if cmd == 'launch':
        subprocess.Popen(['python3', '/home/john/SPECTRA/spectra_gui_launcher.py'])
        return {"status": "launched", "port": 5000}
    elif cmd == 'query':
        # Query SPECTRA DB
        pass
```

### Step 2: Add Button (2K tokens)
```html
<button onclick="cmd('spectra launch')">📡 Launch SPECTRA</button>
```

### Step 3: Database Query (5K tokens)
```python
import sqlite3
def query_spectra_db(query):
    db = sqlite3.connect('/home/john/SPECTRA/spectra.db')
    # Execute query, return results
```

## Recommendation

**Best Approach**: Don't rebuild - just add launcher button!

**Why**: SPECTRA is complete standalone system. Better to:
1. Add "Launch SPECTRA" button (opens port 5000)
2. Add quick query commands
3. Keep both interfaces separate (9876 + 5000)

**Token Cost**: ~5K tokens for launcher button

**Benefit**: Get all SPECTRA features without rebuilding!

## Commands to Add

```
spectra launch          → Start SPECTRA GUI on port 5000
spectra query: keyword  → Quick DB search
telegram: @channel      → Archive channel
spectra stats           → Show SPECTRA database stats
```

**Total Integration**: 5K-10K tokens
**Result**: Full Telegram archiving + database crawling

Want me to implement this minimal integration?
