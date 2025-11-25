# LAT5150 Dynamic Page Integration API

**Complete guide for integrating external pages with full TEMPEST compliance**

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Page Registration](#page-registration)
4. [TEMPEST Compliance](#tempest-compliance)
5. [Tactical Theming Specification](#tactical-theming-specification)
6. [API Reference](#api-reference)
7. [Security Considerations](#security-considerations)
8. [Examples](#examples)

---

## Overview

The LAT5150 Dynamic Page Integration API allows external projects to register custom web pages dynamically with the tactical web UI. All pages inherit the military-grade TEMPEST-compliant design system.

### Key Features

✅ **Dynamic Registration** - No code changes to main UI
✅ **TEMPEST Compliance** - Automatic EMF reduction validation
✅ **Tactical Theming** - Three display modes (Comfort, Day, Night)
✅ **Security Classification** - Visual classification banners
✅ **Localhost-Only** - APT-grade security hardening
✅ **API Endpoints** - Register custom REST APIs
✅ **Persistent Storage** - Pages survive restarts

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│  External Project (e.g., Jina Cyber Retrieval)           │
│                                                           │
│  ┌─────────────────────────────────────────────┐        │
│  │  Page Registration                           │        │
│  │  • HTML content (TEMPEST-compliant)          │        │
│  │  • Metadata (title, category, classification)│        │
│  │  • API endpoints                             │        │
│  │  • Custom styling (validated)                │        │
│  └──────────────────┬───────────────────────────┘        │
│                     │                                     │
└─────────────────────┼─────────────────────────────────────┘
                      │ POST /api/pages/register
                      ▼
┌──────────────────────────────────────────────────────────┐
│  LAT5150 Dynamic Page Registry                           │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Validation  │→ │   Storage    │→ │   Blueprint    │ │
│  │ • Schema    │  │   /opt/      │  │   Creation     │ │
│  │ • TEMPEST   │  │   lat5150/   │  │                │ │
│  │ • Security  │  │   pages/     │  │                │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│                                                           │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Tactical Web UI                                          │
│  • Page rendered at /page/<page_id>                      │
│  • API routes at <custom_route>                          │
│  • Tactical theme applied automatically                  │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Import the API

```python
from dynamic_page_api import (
    get_page_registry,
    PageRegistration,
    PageMetadata,
    PageEndpoint
)

registry = get_page_registry()
```

### 2. Create Your Page

```python
from datetime import datetime

# Define metadata
metadata = PageMetadata(
    page_id="my_custom_page",
    title="My Custom Analysis Tool",
    category="analysis",
    icon="🔍",
    route="/custom-analysis",
    description="Advanced threat analysis dashboard",
    security_classification="CUI",
    tempest_compliant=True,
    registered_by="my_project",
    registered_at=datetime.utcnow().isoformat(),
    version="1.0.0"
)

# Create HTML content (TEMPEST-compliant)
html_content = """
<div class="tactical-container">
    <div class="tactical-section">
        <h2 class="tactical-section-title">Analysis Dashboard</h2>
        <div class="tactical-grid">
            <div class="tactical-card">
                <div class="tactical-card-header">Threat Score</div>
                <div class="tactical-card-value" id="threat-score">--</div>
            </div>
            <div class="tactical-card">
                <div class="tactical-card-header">Active Alerts</div>
                <div class="tactical-card-value" id="active-alerts">--</div>
            </div>
        </div>
    </div>
</div>
"""

# Define API endpoints
endpoints = [
    PageEndpoint(
        method="GET",
        path="/api/custom-analysis/status",
        handler="get_status",
        requires_auth=True,
        rate_limit=100
    ),
    PageEndpoint(
        method="POST",
        path="/api/custom-analysis/analyze",
        handler="run_analysis",
        requires_auth=True,
        rate_limit=10,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 10}
            },
            "required": ["query"]
        }
    )
]

# Create registration
registration = PageRegistration(
    metadata=metadata,
    html_content=html_content,
    endpoints=endpoints
)

# Register the page
registry.register_page(registration)
print(f"✓ Page registered at: {metadata.route}")
```

### 3. Access Your Page

Navigate to: `http://localhost:5001/page/my_custom_page`

---

## Page Registration

### PageMetadata

Complete specification for page metadata:

```python
@dataclass
class PageMetadata:
    page_id: str                    # Unique identifier (Python identifier)
    title: str                      # Display title (shown in header)
    category: str                   # Category (see below)
    icon: str                       # Icon (emoji or class name)
    route: str                      # URL route (starts with /)
    description: str                # Brief description
    security_classification: str    # Classification level
    tempest_compliant: bool         # TEMPEST compliance flag
    registered_by: str              # Project/module name
    registered_at: str              # ISO timestamp
    version: str                    # Page version (semver)
    requires_auth: bool = True      # Requires authentication
    tactical_mode: str = "comfort"  # Default tactical display mode
    custom_css: Optional[str] = None    # Additional CSS (validated)
    custom_js: Optional[str] = None     # Additional JavaScript (sandboxed)
```

#### Field Specifications

**`page_id`** (Required)
- Must be valid Python identifier
- Lowercase with underscores
- Examples: `cyber_retrieval`, `threat_analysis`, `mission_planning`

**`title`** (Required)
- Human-readable display name
- Shown in page header and navigation
- Examples: "Cyber Threat Retrieval", "Mission Planning Console"

**`category`** (Required)
- One of: `"analysis"`, `"operations"`, `"admin"`, `"custom"`, `"integration"`
- Used for grouping in navigation

**`icon`** (Required)
- Emoji: `"🔍"`, `"⚡"`, `"🛡️"`, `"📊"`
- Font Awesome class: `"fa-shield"`, `"fa-chart-line"`
- Custom SVG (inline)

**`route`** (Required)
- URL path starting with `/`
- Lowercase with hyphens
- Examples: `/cyber-retrieval`, `/threat-analysis`

**`description`** (Required)
- Brief description (1-2 sentences)
- Shown in page header

**`security_classification`** (Required)
- One of: `"UNCLASSIFIED"`, `"CUI"`, `"SECRET"`, `"TOP_SECRET"`
- Displays colored banner at top of page

**`tempest_compliant`** (Required)
- `True`: Enforces TEMPEST validation (recommended)
- `False`: Standard mode (not recommended for sensitive work)

**`registered_by`** (Required)
- Your project/module name
- Examples: `"jina_cyber_retrieval"`, `"custom_analytics"`

**`registered_at`** (Required)
- ISO 8601 timestamp
- `datetime.utcnow().isoformat()`

**`version`** (Required)
- Semantic versioning: `"MAJOR.MINOR.PATCH"`
- Example: `"1.0.0"`, `"2.1.3"`

### PageEndpoint

API endpoint specification:

```python
@dataclass
class PageEndpoint:
    method: str                 # HTTP method
    path: str                   # API path
    handler: str                # Handler function name
    requires_auth: bool = True  # Authentication required
    rate_limit: int = 100       # Requests per minute
    input_schema: Optional[Dict] = None  # JSON schema validation
```

#### Field Specifications

**`method`** (Required)
- HTTP method: `"GET"`, `"POST"`, `"PUT"`, `"DELETE"`

**`path`** (Required)
- API endpoint path
- Must start with `/api/`
- Examples: `/api/cyber-retrieval/search`, `/api/analysis/run`

**`handler`** (Required)
- Function name that handles the request
- Must be implemented by your project

**`requires_auth`** (Optional, default: `True`)
- Whether endpoint requires authentication

**`rate_limit`** (Optional, default: `100`)
- Maximum requests per minute per client

**`input_schema`** (Optional)
- JSON Schema for request validation
- See: https://json-schema.org/

---

## TEMPEST Compliance

**TEMPEST** (Telecommunications Electronics Material Protected from Emanating Spurious Transmissions) compliance reduces electromagnetic emissions that can be intercepted.

### Why TEMPEST Matters

- **OPSEC**: Prevents remote monitoring of screen content
- **Security**: Reduces EM signature of sensitive operations
- **Classification**: Required for SECRET and above

### TEMPEST Requirements

When `tempest_compliant=True`, your page must follow these rules:

#### ❌ **Forbidden**

```css
/* Animations increase EMF emissions */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.element {
    animation: pulse 2s infinite;  /* ❌ FORBIDDEN */
    transition: all 0.3s ease;     /* ❌ FORBIDDEN */
}

/* External resources violate OPSEC */
<link href="https://cdn.example.com/style.css"> /* ❌ FORBIDDEN */
<script src="http://example.com/lib.js">         /* ❌ FORBIDDEN */
<img src="https://tracking.com/pixel.png">       /* ❌ FORBIDDEN */
```

#### ✅ **Allowed**

```css
/* Instant state changes (no animation) */
.element {
    opacity: 1;
}

.element:hover {
    opacity: 0.8;  /* Instant change ✓ */
}

/* Limited transitions on specific properties */
.button {
    transition: opacity 0ms;  /* Instant ✓ */
    /* OR */
    transition: opacity 50ms; /* Minimal (50ms max) ✓ */
}

/* Tactical color changes (instant) */
.status-active {
    background: var(--status-success);  /* ✓ */
    color: var(--text-inverse);         /* ✓ */
}
```

### TEMPEST Validation

The registry automatically validates:

1. **No animations** - No `@keyframes` or long `animation` rules
2. **No external resources** - All assets must be inline
3. **Brightness limits** - Maximum 85% brightness
4. **Color compliance** - Must use tactical color variables

---

## Tactical Theming Specification

All pages inherit the LAT5150 tactical design system with three display modes.

### Display Modes

#### 1. **Comfort Mode** (Default - NATO SDIP-27 Level C)

Eye-friendly for extended operations, unclassified sensitive.

```css
--bg-primary: #1a2228;      /* Dark blue-grey */
--text-primary: #d4e4f0;    /* Light blue-white */
--tactical-primary: #4db8e8; /* Bright blue */
--status-success: #5fdc8f;  /* Green */
--status-error: #ff6b6b;    /* Red */
```

**Use Case**: Extended monitoring, analysis, development

#### 2. **Day Mode**

High contrast for outdoor/bright conditions.

```css
--bg-primary: #1a1a1a;      /* Pure dark grey */
--text-primary: #e0e0e0;    /* Light grey */
--tactical-primary: #00ff00; /* Bright green */
--status-success: #00ff00;  /* Green */
--status-error: #ff0000;    /* Red */
```

**Use Case**: Bright environments, sunlight, field operations

#### 3. **Night Mode**

Reduced brightness for low-light operations (red tint).

```css
--bg-primary: #0a0a0a;      /* Very dark */
--text-primary: #ff6666;    /* Dim red */
--tactical-primary: #ff4444; /* Red */
--status-success: #ff6666;  /* Dim red */
--status-error: #ff3333;    /* Bright red */
```

**Use Case**: Night operations, low-light, stealth mode

### Color Variables

Use these CSS variables for automatic theme switching:

```css
/* Backgrounds */
--bg-primary        /* Main background */
--bg-secondary      /* Secondary background */
--bg-tertiary       /* Card/panel background */
--bg-input          /* Input field background */
--bg-hover          /* Hover state */
--bg-active         /* Active state */

/* Text */
--text-primary      /* Primary text */
--text-secondary    /* Secondary text */
--text-muted        /* Muted text */
--text-inverse      /* Inverse text (on colored backgrounds) */

/* Borders */
--border-primary    /* Primary border */
--border-secondary  /* Secondary border */
--border-active     /* Active border */

/* Status Colors */
--status-success    /* Success (green/red) */
--status-warning    /* Warning (yellow/orange) */
--status-error      /* Error (red) */
--status-info       /* Info (blue/cyan) */
--status-processing /* Processing (orange) */

/* Tactical Accents */
--tactical-primary   /* Primary accent */
--tactical-secondary /* Secondary accent */
--tactical-accent    /* Highlight accent */

/* Effects */
--shadow-tactical   /* Tactical shadow */
--glow-tactical     /* Tactical glow */
```

### Tactical Components

Pre-built components that automatically adapt to tactical modes:

#### Container

```html
<div class="tactical-container">
    <!-- Your content -->
</div>
```

#### Section

```html
<div class="tactical-section">
    <h2 class="tactical-section-title">Section Title</h2>
    <div class="tactical-section-content">
        <!-- Content -->
    </div>
</div>
```

#### Grid Layout

```html
<div class="tactical-grid">
    <div class="tactical-card">
        <div class="tactical-card-header">Title</div>
        <div class="tactical-card-value">Value</div>
        <div class="tactical-card-footer">Footer</div>
    </div>
    <!-- More cards -->
</div>
```

#### Data Display

```html
<div class="tactical-stat">
    <span class="tactical-stat-label">CPU Usage</span>
    <span class="tactical-stat-value">45%</span>
    <span class="tactical-stat-unit">percent</span>
</div>
```

#### Status Indicators

```html
<span class="tactical-status tactical-status-success">OPERATIONAL</span>
<span class="tactical-status tactical-status-warning">DEGRADED</span>
<span class="tactical-status tactical-status-error">OFFLINE</span>
<span class="tactical-status tactical-status-info">STANDBY</span>
```

#### Buttons

```html
<button class="tactical-btn tactical-btn-primary">Primary Action</button>
<button class="tactical-btn tactical-btn-secondary">Secondary</button>
<button class="tactical-btn tactical-btn-danger">Danger</button>
```

#### Code Blocks

```html
<pre class="tactical-code"><code>
{
    "status": "operational",
    "systems": 84
}
</code></pre>
```

---

## API Reference

### Registry Methods

```python
from dynamic_page_api import get_page_registry

registry = get_page_registry()
```

#### `register_page(registration, overwrite=False) -> bool`

Register a new page.

**Parameters:**
- `registration` (PageRegistration): Complete page registration
- `overwrite` (bool): Allow replacing existing page

**Returns:**
- `bool`: Success status

**Raises:**
- `ValueError`: Invalid registration data
- `PermissionError`: Page exists and overwrite=False

**Example:**

```python
success = registry.register_page(registration)
if success:
    print("Page registered successfully")
```

#### `unregister_page(page_id) -> bool`

Remove a registered page.

**Parameters:**
- `page_id` (str): Page identifier

**Returns:**
- `bool`: Success status

#### `get_page(page_id) -> Optional[PageRegistration]`

Get page registration by ID.

**Parameters:**
- `page_id` (str): Page identifier

**Returns:**
- `PageRegistration` or `None`

#### `list_pages(category=None) -> List[PageMetadata]`

List all registered pages.

**Parameters:**
- `category` (str, optional): Filter by category

**Returns:**
- `List[PageMetadata]`: Sorted list of page metadata

**Example:**

```python
# List all pages
all_pages = registry.list_pages()

# List only analysis pages
analysis_pages = registry.list_pages(category="analysis")

for page in analysis_pages:
    print(f"- {page.title} ({page.page_id})")
```

#### `get_page_html(page_id, tactical_mode="comfort") -> Optional[str]`

Get rendered HTML with tactical theme.

**Parameters:**
- `page_id` (str): Page identifier
- `tactical_mode` (str): Display mode ("comfort", "day", "night")

**Returns:**
- `str`: Rendered HTML or `None`

### REST API Endpoints

#### `GET /api/pages`

List all registered pages.

**Query Parameters:**
- `category` (optional): Filter by category

**Response:**

```json
{
    "pages": [
        {
            "page_id": "cyber_retrieval",
            "title": "Cyber Threat Retrieval",
            "category": "analysis",
            "route": "/cyber-retrieval",
            "security_classification": "CUI",
            "tempest_compliant": true
        }
    ],
    "total": 1
}
```

#### `GET /api/pages/<page_id>`

Get page details.

**Response:**

```json
{
    "metadata": {
        "page_id": "cyber_retrieval",
        "title": "Cyber Threat Retrieval",
        ...
    },
    "endpoints": [
        {
            "method": "GET",
            "path": "/api/cyber-retrieval/search",
            "handler": "search_threats"
        }
    ],
    "has_initialization_script": false
}
```

#### `POST /api/pages/register`

Register a new page.

**Request Body:**

```json
{
    "metadata": {
        "page_id": "my_page",
        "title": "My Custom Page",
        ...
    },
    "html_content": "<div>...</div>",
    "endpoints": [...],
    "overwrite": false
}
```

**Response:**

```json
{
    "success": true,
    "page_id": "my_page",
    "route": "/my-page"
}
```

#### `DELETE /api/pages/<page_id>`

Unregister a page.

**Response:**

```json
{
    "success": true,
    "page_id": "my_page"
}
```

#### `GET /api/pages/registry-info`

Get registry statistics.

**Response:**

```json
{
    "total_pages": 5,
    "pages_by_category": {
        "analysis": 2,
        "operations": 1,
        "admin": 1,
        "custom": 1
    },
    "tempest_compliant": 4,
    "storage_path": "/opt/lat5150/pages"
}
```

#### `GET /page/<page_id>`

Render a registered page.

**Query Parameters:**
- `mode` (optional): Tactical mode ("comfort", "day", "night")

**Response:** HTML page with tactical theme

---

## Security Considerations

### Localhost-Only Access

All pages inherit APT-grade security:

- **Localhost binding**: 127.0.0.1 only
- **Token authentication**: Bearer tokens required
- **Rate limiting**: Configurable per endpoint
- **Input validation**: JSON schema enforcement
- **Audit logging**: All requests logged

### Sandboxed Execution

JavaScript in pages runs in sandboxed context:

- No eval() or Function()
- No inline event handlers
- Content Security Policy enforced
- No external script loading

### File System Protection

Page storage is isolated:

```
/opt/lat5150/pages/
├── cyber_retrieval.json
├── threat_analysis.json
└── custom_dashboard.json
```

Only authorized processes can write to this directory.

---

## Examples

### Example 1: Simple Status Dashboard

```python
from dynamic_page_api import *
from datetime import datetime

metadata = PageMetadata(
    page_id="system_status",
    title="System Status Dashboard",
    category="operations",
    icon="⚡",
    route="/system-status",
    description="Real-time system health monitoring",
    security_classification="UNCLASSIFIED",
    tempest_compliant=True,
    registered_by="monitoring_module",
    registered_at=datetime.utcnow().isoformat(),
    version="1.0.0"
)

html_content = """
<div class="tactical-container">
    <div class="tactical-grid">
        <div class="tactical-card">
            <div class="tactical-card-header">AI Acceleration</div>
            <div class="tactical-card-value" id="ai-tops">100 TOPS</div>
            <span class="tactical-status tactical-status-success">OPERATIONAL</span>
        </div>
        <div class="tactical-card">
            <div class="tactical-card-header">DSMIL Devices</div>
            <div class="tactical-card-value" id="dsmil-count">79 SAFE</div>
            <span class="tactical-status tactical-status-success">SECURE</span>
        </div>
        <div class="tactical-card">
            <div class="tactical-card-header">Cognitive Memory</div>
            <div class="tactical-card-value" id="memory-entries">1,247</div>
            <span class="tactical-status tactical-status-info">ACTIVE</span>
        </div>
    </div>
</div>
"""

endpoints = [
    PageEndpoint(
        method="GET",
        path="/api/system-status/current",
        handler="get_current_status"
    )
]

registration = PageRegistration(
    metadata=metadata,
    html_content=html_content,
    endpoints=endpoints
)

registry = get_page_registry()
registry.register_page(registration)
```

### Example 2: Cyber Threat Analysis

```python
metadata = PageMetadata(
    page_id="cyber_threats",
    title="Cyber Threat Analysis",
    category="analysis",
    icon="🛡️",
    route="/cyber-threats",
    description="Advanced persistent threat detection and analysis",
    security_classification="SECRET",
    tempest_compliant=True,
    registered_by="cyber_defense_module",
    registered_at=datetime.utcnow().isoformat(),
    version="2.1.0"
)

html_content = """
<div class="tactical-container">
    <div class="tactical-section">
        <h2 class="tactical-section-title">Threat Intelligence</h2>

        <div class="tactical-code-block">
            <div class="tactical-code-header">Latest Threat Indicators</div>
            <pre class="tactical-code" id="threat-indicators">
Loading threat data...
            </pre>
        </div>

        <div class="tactical-grid">
            <div class="tactical-stat">
                <span class="tactical-stat-label">Active Threats</span>
                <span class="tactical-stat-value" id="active-threats">--</span>
            </div>
            <div class="tactical-stat">
                <span class="tactical-stat-label">Quarantined</span>
                <span class="tactical-stat-value" id="quarantined">--</span>
            </div>
            <div class="tactical-stat">
                <span class="tactical-stat-label">Risk Score</span>
                <span class="tactical-stat-value" id="risk-score">--</span>
            </div>
        </div>
    </div>
</div>

<script>
// Initialization (sandboxed)
(function() {
    function updateStatus() {
        fetch('/api/cyber-threats/status')
            .then(r => r.json())
            .then(data => {
                document.getElementById('active-threats').textContent = data.active;
                document.getElementById('quarantined').textContent = data.quarantined;
                document.getElementById('risk-score').textContent = data.risk_score;
            });
    }

    updateStatus();
    setInterval(updateStatus, 5000); // Update every 5 seconds
})();
</script>
"""

endpoints = [
    PageEndpoint(
        method="GET",
        path="/api/cyber-threats/status",
        handler="get_threat_status",
        rate_limit=60
    ),
    PageEndpoint(
        method="POST",
        path="/api/cyber-threats/analyze",
        handler="analyze_threat",
        rate_limit=10,
        input_schema={
            "type": "object",
            "properties": {
                "indicator": {"type": "string"},
                "type": {"enum": ["ip", "domain", "hash", "url"]}
            },
            "required": ["indicator", "type"]
        }
    )
]

registration = PageRegistration(
    metadata=metadata,
    html_content=html_content,
    endpoints=endpoints
)

registry.register_page(registration)
```

### Example 3: Custom Analytics with External Data

```python
metadata = PageMetadata(
    page_id="custom_analytics",
    title="Custom Analytics Engine",
    category="custom",
    icon="📊",
    route="/custom-analytics",
    description="Customizable data analysis and visualization",
    security_classification="CUI",
    tempest_compliant=True,
    registered_by="analytics_framework",
    registered_at=datetime.utcnow().isoformat(),
    version="1.5.2",
    custom_css="""
    .analytics-chart {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        padding: var(--spacing-md);
        border-radius: 4px;
    }

    .analytics-metric {
        display: flex;
        justify-content: space-between;
        padding: var(--spacing-sm);
        border-bottom: 1px solid var(--border-secondary);
    }
    """
)

html_content = """
<div class="tactical-container">
    <div class="tactical-section">
        <h2 class="tactical-section-title">Analytics Overview</h2>

        <div class="analytics-chart">
            <canvas id="main-chart" width="800" height="400"></canvas>
        </div>

        <div class="tactical-section">
            <h3 class="tactical-section-subtitle">Key Metrics</h3>
            <div id="metrics-container">
                <!-- Populated by JavaScript -->
            </div>
        </div>
    </div>
</div>
"""

registration = PageRegistration(
    metadata=metadata,
    html_content=html_content,
    endpoints=[
        PageEndpoint(
            method="GET",
            path="/api/analytics/metrics",
            handler="get_metrics"
        )
    ]
)

registry.register_page(registration)
```

---

## Best Practices

### 1. Follow TEMPEST Guidelines

- ✅ Use instant state changes (no animations)
- ✅ Use tactical color variables
- ✅ Inline all resources
- ✅ Validate before registration

### 2. Use Tactical Components

- ✅ Use `.tactical-*` classes for consistency
- ✅ Follow grid system (8px units)
- ✅ Use monospace fonts for data

### 3. Security First

- ✅ Set appropriate classification
- ✅ Validate all inputs
- ✅ Use rate limiting
- ✅ Log all operations

### 4. Performance

- ✅ Minimize DOM updates
- ✅ Use efficient selectors
- ✅ Cache API responses
- ✅ Lazy load heavy content

### 5. Accessibility

- ✅ High contrast in all modes
- ✅ Keyboard navigation
- ✅ Screen reader compatible
- ✅ Clear visual hierarchy

---

## Troubleshooting

### Page Not Registering

**Error**: `ValueError: Invalid page_id`

**Solution**: Ensure `page_id` is a valid Python identifier (lowercase, underscores only)

```python
# ❌ Wrong
page_id = "my-page"
page_id = "My Page"
page_id = "my.page"

# ✓ Correct
page_id = "my_page"
page_id = "custom_dashboard"
```

### TEMPEST Compliance Violation

**Error**: `ValueError: TEMPEST compliance violations: animation: Animations increase EMF emissions`

**Solution**: Remove animations and transitions

```css
/* ❌ Wrong */
.element {
    transition: all 0.3s ease;
    animation: fadeIn 1s;
}

/* ✓ Correct */
.element {
    /* Instant changes only */
    opacity: 1;
}

.element:hover {
    opacity: 0.8;  /* Instant */
}
```

### Page Not Appearing

**Error**: Page registered but not showing

**Solution**: Ensure Flask blueprint is registered

```python
from flask import Flask

app = Flask(__name__)

# Register the page
registry.register_page(registration)

# Create and register blueprint
blueprint = registry.create_blueprint(metadata.page_id, app)
app.register_blueprint(blueprint)

# OR use the helper function
from dynamic_page_api import register_page_api_routes
register_page_api_routes(app)
```

---

## Support

For issues or questions:

1. Check this documentation
2. Review examples above
3. Check logs: `/var/log/lat5150/pages.log`
4. Review TEMPEST compliance guide
5. Contact LAT5150 development team

---

## Version History

- **v1.0.0** (2025-01-13) - Initial release
  - Dynamic page registration
  - TEMPEST compliance validation
  - Three tactical display modes
  - REST API for page management
  - Persistent storage
  - Security classification banners
