# DSMIL AI Interface Options

Three interface choices - pick what works for you:

---

## 1. Complete Modern TUI ⭐ RECOMMENDED

**File:** `ai_tui_complete.py`

✅ **ALL features** from original
✅ **Clean modern design**
✅ **Single-key shortcuts**
✅ **Organized submenus**

### Features

- Query AI (5 models)
- Models management
- RAG Knowledge (add/search/list/stats)
- Guardrails (6 presets + custom)
- Test Models
- Settings (export/import config)
- System Info

### Usage

```bash
./ai_tui_complete.py
```

---

## 2. Minimal Modern TUI

**File:** `ai_tui_v2.py`

✅ **Core features only**
✅ **Ultra-clean design**
✅ **Fast navigation**

### Features

- Query AI
- View Models
- RAG basics (add/search/list)
- Guardrails view
- System status

### Usage

```bash
./ai_tui_v2.py
```

---

## 3. Original TUI

**File:** `ai_tui.py`

✅ **All features**
❌ **Heavy boxes**
❌ **Number navigation**
❌ **Cluttered**

Still available if you prefer the old style.

---

## 4. Clean CLI

**File:** `ai.py`

**Fast command-line queries**

```bash
ai "question"
ai -f "fast question"
ai -c "write code"
ai -i  # interactive
```

---

## Comparison

| Feature | Complete | Minimal | Original | CLI |
|---------|----------|---------|----------|-----|
| Query AI | ✓ | ✓ | ✓ | ✓ |
| RAG Full | ✓ | Basic | ✓ | ✗ |
| Guardrails | 6+ | View | 6+ | ✗ |
| Test Models | ✓ | ✗ | ✓ | ✗ |
| Settings | ✓ | ✗ | ✓ | ✗ |
| Clean Design | ✓ | ✓ | ✗ | ✓ |
| Single-key Nav | ✓ | ✓ | ✗ | N/A |

---

## Recommendation

**Use `ai_tui_complete.py`** - It has everything, organized cleanly.

**Use `ai.py`** for quick command-line queries.

---

## Quick Start

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine

# TUI (recommended)
./ai_tui_complete.py

# CLI
./ai.py "your question"
```

Enjoy!
