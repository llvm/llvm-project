# LAT5150 Web Interface - Dynamic Page Integration

**Extensible tactical web UI with TEMPEST-compliant dynamic page registration**

## Overview

The LAT5150 web interface provides a dynamic page integration system that allows external projects to register custom pages without modifying the main UI code. All pages inherit the military-grade TEMPEST-compliant design system with three tactical display modes.

## Quick Start

### 1. Register Your Page

```python
from dynamic_page_api import get_page_registry, PageRegistration, PageMetadata, PageEndpoint
from datetime import datetime

# Create your page
metadata = PageMetadata(
    page_id="my_page",
    title="My Custom Page",
    category="analysis",
    icon="🔍",
    route="/my-page",
    description="Custom analysis tool",
    security_classification="CUI",
    tempest_compliant=True,
    registered_by="my_project",
    registered_at=datetime.utcnow().isoformat(),
    version="1.0.0"
)

html_content = """
<div class="tactical-container">
    <div class="tactical-section">
        <h2 class="tactical-section-title">My Analysis</h2>
        <!-- Your TEMPEST-compliant HTML -->
    </div>
</div>
"""

registration = PageRegistration(
    metadata=metadata,
    html_content=html_content,
    endpoints=[]
)

# Register it
registry = get_page_registry()
registry.register_page(registration)
```

### 2. Access Your Page

Navigate to: `http://localhost:5001/page/my_page`

## Documentation

### 📚 Complete Guides

- **[DYNAMIC_PAGE_INTEGRATION.md](./DYNAMIC_PAGE_INTEGRATION.md)** - Complete integration guide
  - Page registration
  - TEMPEST compliance requirements
  - Tactical theming specification
  - API reference
  - Security considerations
  - Full examples

- **[TACTICAL_THEME_REFERENCE.css](./TACTICAL_THEME_REFERENCE.css)** - CSS reference
  - All available CSS classes
  - Color variables for all modes
  - Component library
  - TEMPEST guidelines
  - Utility classes

- **[example_page_integration.py](./example_page_integration.py)** - Working examples
  - Cyber threat retrieval page
  - Simple dashboard
  - Ready to run

## Features

### ✅ **Dynamic Registration**

No code changes to main UI required. Pages are registered programmatically and persist across restarts.

```python
# Register once
registry.register_page(registration)

# Available immediately
# Survives server restarts (persisted to /opt/lat5150/pages/)
```

### ✅ **TEMPEST Compliance**

Automatic validation of TEMPEST requirements:

- No animations (EMF reduction)
- No external resources (OPSEC)
- Brightness limits enforced
- Tactical color schemes only

```python
metadata.tempest_compliant = True  # Enables validation
```

### ✅ **Three Tactical Modes**

All pages automatically support three display modes:

| Mode | Use Case | Colors |
|------|----------|--------|
| **Comfort** | Extended operations | Blue-grey (NATO SDIP-27 Level C) |
| **Day** | Bright environments | High contrast green |
| **Night** | Low-light operations | Red tint for night vision |

Users switch modes with one click. All tactical CSS variables update automatically.

### ✅ **Security Classification**

Visual classification banners at top of every page:

```python
security_classification="CUI"  # Shows yellow banner
# Options: UNCLASSIFIED, CUI, SECRET, TOP_SECRET
```

### ✅ **API Endpoints**

Register custom REST APIs with your page:

```python
endpoints = [
    PageEndpoint(
        method="POST",
        path="/api/my-page/action",
        handler="handle_action",
        rate_limit=30,
        input_schema={...}
    )
]
```

### ✅ **Persistent Storage**

Pages are stored in `/opt/lat5150/pages/` and survive server restarts:

```
/opt/lat5150/pages/
├── cyber_threat_retrieval.json
├── simple_dashboard.json
└── my_custom_page.json
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  External Project                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │  Page Registration                              │ │
│  │  • Metadata                                     │ │
│  │  • HTML (TEMPEST-compliant)                     │ │
│  │  • API endpoints                                │ │
│  └──────────────┬─────────────────────────────────┘ │
└─────────────────┼───────────────────────────────────┘
                  │ POST /api/pages/register
                  ▼
┌──────────────────────────────────────────────────────┐
│  Dynamic Page Registry                                │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ Validate │→ │  Store   │→ │  Create Blueprint  │ │
│  │ • Schema │  │ JSON file│  │  Flask routes      │ │
│  │ • TEMPEST│  │ Persist  │  │  API endpoints     │ │
│  └──────────┘  └──────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│  Tactical Web UI (Flask)                              │
│  • Renders page at /page/<page_id>                   │
│  • Tactical theme applied                            │
│  • Security classification banner                    │
│  • Three display modes                               │
└──────────────────────────────────────────────────────┘
```

## File Structure

```
03-web-interface/
├── dynamic_page_api.py                  # Core API (600 lines)
├── DYNAMIC_PAGE_INTEGRATION.md          # Complete guide (1000+ lines)
├── TACTICAL_THEME_REFERENCE.css         # CSS reference (600 lines)
├── example_page_integration.py          # Working examples
├── README_DYNAMIC_PAGES.md              # This file
├── unified_tactical_api.py              # Main Flask app
├── secured_self_coding_api.py           # Security layer
└── tactical_self_coding_ui.html         # Base UI template
```

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/pages` | List all registered pages |
| `GET` | `/api/pages/<page_id>` | Get page details |
| `POST` | `/api/pages/register` | Register new page |
| `DELETE` | `/api/pages/<page_id>` | Unregister page |
| `GET` | `/api/pages/registry-info` | Registry statistics |
| `GET` | `/page/<page_id>?mode=comfort` | Render page |

### Python API

```python
from dynamic_page_api import get_page_registry

registry = get_page_registry()

# Register page
registry.register_page(registration, overwrite=False)

# Get page
page = registry.get_page("page_id")

# List pages
pages = registry.list_pages(category="analysis")

# Get rendered HTML
html = registry.get_page_html("page_id", tactical_mode="comfort")

# Create Flask blueprint
blueprint = registry.create_blueprint("page_id", app)
app.register_blueprint(blueprint)
```

## TEMPEST Compliance

### What is TEMPEST?

**TEMPEST** (Telecommunications Electronics Material Protected from Emanating Spurious Transmissions) is a standard for reducing electromagnetic emissions that can be intercepted remotely.

### Why It Matters

- **OPSEC**: Prevents remote monitoring of screen content
- **Security**: Reduces EM signature of sensitive operations
- **Classification**: Required for SECRET and above

### Requirements

#### ❌ **Forbidden**

```css
/* Animations increase EMF emissions */
@keyframes pulse { ... }  /* ❌ */
.element { animation: pulse 2s; }  /* ❌ */
.element { transition: all 0.3s; }  /* ❌ */

/* External resources violate OPSEC */
<script src="https://cdn.example.com/lib.js">  /* ❌ */
<link href="http://example.com/style.css">  /* ❌ */
```

#### ✅ **Allowed**

```css
/* Instant state changes */
.element { opacity: 1; }
.element:hover { opacity: 0.8; }  /* ✓ Instant */

/* Tactical color variables */
.element { color: var(--tactical-primary); }  /* ✓ */
```

### Validation

When `tempest_compliant=True`:

1. ✅ No `@keyframes` or `animation`
2. ✅ No long `transition` durations
3. ✅ No external resources (http://, https://)
4. ✅ Brightness limits enforced
5. ✅ Tactical color variables only

Violations trigger `ValueError` during registration.

## Tactical Theming

### Display Modes

All pages automatically support three modes via CSS variables:

#### 1. **Comfort Mode** (Default)

NATO SDIP-27 Level C protection. Eye-friendly for extended operations.

```css
--bg-primary: #1a2228;
--text-primary: #d4e4f0;
--tactical-primary: #4db8e8;
```

#### 2. **Day Mode**

High contrast for outdoor/bright conditions.

```css
--bg-primary: #1a1a1a;
--text-primary: #e0e0e0;
--tactical-primary: #00ff00;
```

#### 3. **Night Mode**

Reduced brightness for low-light operations.

```css
--bg-primary: #0a0a0a;
--text-primary: #ff6666;
--tactical-primary: #ff4444;
```

### CSS Variables

```css
/* Use these in your HTML */
--bg-primary, --bg-secondary, --bg-tertiary
--text-primary, --text-secondary, --text-muted
--border-primary, --border-secondary, --border-active
--status-success, --status-warning, --status-error, --status-info
--tactical-primary, --tactical-secondary, --tactical-accent
```

### Pre-Built Components

```html
<!-- Containers -->
<div class="tactical-container">...</div>
<div class="tactical-section">...</div>
<div class="tactical-grid">...</div>

<!-- Cards -->
<div class="tactical-card">
    <div class="tactical-card-header">Title</div>
    <div class="tactical-card-value">42</div>
    <div class="tactical-card-footer">Footer</div>
</div>

<!-- Status -->
<span class="tactical-status tactical-status-success">OPERATIONAL</span>
<span class="tactical-status tactical-status-error">OFFLINE</span>

<!-- Buttons -->
<button class="tactical-btn tactical-btn-primary">Action</button>
<button class="tactical-btn tactical-btn-secondary">Cancel</button>

<!-- Data Display -->
<div class="tactical-stat">
    <span class="tactical-stat-label">CPU Usage</span>
    <span class="tactical-stat-value">45%</span>
</div>

<!-- Code -->
<pre class="tactical-code"><code>{"status": "ok"}</code></pre>

<!-- Forms -->
<label class="tactical-label">Query</label>
<input type="text" class="tactical-input" />
<select class="tactical-select">...</select>

<!-- Tables -->
<table class="tactical-table">
    <thead><tr><th>Column</th></tr></thead>
    <tbody><tr><td>Data</td></tr></tbody>
</table>

<!-- Alerts -->
<div class="tactical-alert tactical-alert-success">Success message</div>
<div class="tactical-alert tactical-alert-error">Error message</div>
```

## Examples

### Example 1: Simple Status Page

```python
metadata = PageMetadata(
    page_id="system_status",
    title="System Status",
    category="operations",
    icon="⚡",
    route="/status",
    description="System health monitoring",
    security_classification="UNCLASSIFIED",
    tempest_compliant=True,
    registered_by="monitoring",
    registered_at=datetime.utcnow().isoformat(),
    version="1.0.0"
)

html_content = """
<div class="tactical-container">
    <div class="tactical-grid">
        <div class="tactical-card">
            <div class="tactical-card-header">AI Acceleration</div>
            <div class="tactical-card-value">100 TOPS</div>
            <span class="tactical-status tactical-status-success">OPERATIONAL</span>
        </div>
    </div>
</div>
"""

registration = PageRegistration(metadata=metadata, html_content=html_content, endpoints=[])
get_page_registry().register_page(registration)
```

### Example 2: Interactive Analysis Page

See [example_page_integration.py](./example_page_integration.py) for a complete working example with:

- Search interface
- API integration
- Real-time updates
- Result display
- History tracking

## Security

### Localhost-Only

All pages inherit APT-grade security from the main UI:

- 127.0.0.1 binding only
- Token authentication
- Rate limiting
- Input validation
- Audit logging

### Sandboxed Execution

JavaScript runs in sandboxed context:

- No eval() or Function()
- No inline event handlers
- Content Security Policy
- No external script loading

### File System Protection

Pages stored in protected directory:

```bash
/opt/lat5150/pages/
# Only authorized processes can write
```

## Testing

### Run Examples

```bash
cd /home/user/LAT5150DRVMIL/03-web-interface

# Register example pages
python3 example_page_integration.py

# Output:
#   ✓ Successfully registered: Cyber Threat Retrieval
#   ✓ Registered: System Dashboard
```

### Access Pages

```bash
# View all pages
curl http://localhost:5001/api/pages

# View specific page
curl http://localhost:5001/api/pages/cyber_threat_retrieval

# Render page
curl http://localhost:5001/page/cyber_threat_retrieval?mode=comfort
```

### Integration with Flask

```python
from flask import Flask
from dynamic_page_api import register_page_api_routes, get_page_registry

app = Flask(__name__)

# Register page management APIs
register_page_api_routes(app)

# Register your pages
registry = get_page_registry()
# ... register pages ...

# Create blueprints for page endpoints
for page_id in registry.pages.keys():
    blueprint = registry.create_blueprint(page_id, app)
    if blueprint:
        app.register_blueprint(blueprint)

app.run(host='127.0.0.1', port=5001)
```

## Best Practices

### 1. **Always Validate TEMPEST**

```python
metadata.tempest_compliant = True  # ✓ Recommended
# Registry will validate your HTML/CSS
```

### 2. **Use Tactical Components**

```html
<!-- ✓ Good: Uses tactical classes -->
<div class="tactical-card">
    <div class="tactical-card-value">42</div>
</div>

<!-- ✗ Bad: Custom styling -->
<div style="background: #fff; color: #000;">42</div>
```

### 3. **Proper Classification**

```python
# Match your data classification
security_classification="CUI"  # If handling CUI data
security_classification="SECRET"  # If handling SECRET data
```

### 4. **Versioning**

```python
version="1.0.0"  # Semantic versioning
# Major.Minor.Patch
```

### 5. **Error Handling**

```python
try:
    registry.register_page(registration)
except ValueError as e:
    print(f"Validation error: {e}")
except PermissionError as e:
    print(f"Permission error: {e}")
```

## Troubleshooting

### "Page already exists"

Use `overwrite=True`:

```python
registry.register_page(registration, overwrite=True)
```

### "TEMPEST compliance violations"

Check for forbidden patterns:

```python
# ❌ Remove animations
animation: pulse 2s;

# ❌ Remove external resources
<script src="https://...">

# ✓ Use instant transitions
opacity: 1;  /* No transition */
```

### "Page not appearing"

Ensure blueprint is registered:

```python
blueprint = registry.create_blueprint(page_id, app)
app.register_blueprint(blueprint)
```

## Support

**Documentation:**
- DYNAMIC_PAGE_INTEGRATION.md - Complete guide
- TACTICAL_THEME_REFERENCE.css - CSS reference
- example_page_integration.py - Working examples

**Logs:**
- `/var/log/lat5150/pages.log`

**Development Team:**
- LAT5150 DRVMIL Project

---

## Version History

- **v1.0.0** (2025-01-13)
  - Initial release
  - Dynamic page registration
  - TEMPEST compliance validation
  - Three tactical display modes
  - Security classification banners
  - Persistent storage
  - REST API
  - Complete documentation
