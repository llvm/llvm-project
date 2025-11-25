# HERETIC Integration with DSMIL TEMPEST Web Interface

**Complete guide for integrating Heretic abliteration into the DSMIL TEMPEST dashboard**

---

## INTEGRATION METHODS

You have **TWO OPTIONS** for integrating Heretic:

### Option 1: Hook System Integration (Recommended)
✅ Minimal changes to existing code
✅ Automatic refusal detection and monitoring
✅ Seamless integration with existing workflows
✅ Plugin-style architecture

### Option 2: Web API Integration
✅ Full REST API for all Heretic operations
✅ Dedicated web panel with UI
✅ Manual control over abliteration workflows
✅ Job tracking and status monitoring

**You can use BOTH together** for maximum flexibility!

---

## OPTION 1: HOOK SYSTEM INTEGRATION

### Step 1: Register Heretic Hooks

Add to `ai_gui_dashboard.py` in the `initialize_components()` function:

```python
def initialize_components():
    """Initialize AI components"""
    global integrator, security, hephaestus, ...

    # ... existing initialization code ...

    # Initialize Heretic hooks (ADD THIS)
    try:
        from heretic_hook import register_heretic_hooks
        from hook_system import HookManager, create_default_hooks

        # Create or get hook manager
        hook_manager = create_default_hooks()

        # Register Heretic hooks
        register_heretic_hooks(
            hook_manager,
            enable_refusal_detection=True,      # Detect refusals in responses
            enable_safety_monitor=True,         # Track refusal rates
            enable_abliteration_trigger=True,   # Suggest abliteration
            auto_abliterate=False,              # Don't auto-trigger (safety)
            refusal_threshold=0.5               # 50% refusal rate threshold
        )

        print("✓ Heretic hooks registered")
    except Exception as e:
        print(f"Warning: Could not initialize Heretic hooks: {e}")
```

### Step 2: That's It!

Heretic is now integrated! The hooks will:

1. **Automatically detect refusals** in model responses
2. **Monitor safety metrics** (refusal rates over time)
3. **Suggest abliteration** when refusal rate exceeds threshold
4. **Provide data** to TEMPEST dashboard

### What You Get

#### Automatic Refusal Detection
Every response is analyzed for refusal markers:
```
"I'm sorry, I can't help with that" → REFUSAL DETECTED
"Here's the code you requested..." → OK
```

#### Safety Monitoring Dashboard
Access safety metrics via hook metadata:
```python
# In your dashboard route
context_metadata = hook_manager.get_metadata()
safety_metrics = context_metadata.get("heretic", {}).get("safety_metrics", {})

# Returns:
{
    "refusal_count": 15,
    "total_queries": 100,
    "refusal_rate": 0.15,  # 15%
    "recent_refusals": 3    # In last hour
}
```

#### Alert System
When refusal rate exceeds threshold:
```
⚠️ High refusal rate detected: 52.3% (threshold: 50.0%)
   Abliteration recommended
```

---

## OPTION 2: WEB API INTEGRATION

### Step 1: Register Web Routes

Add to `ai_gui_dashboard.py`:

```python
from heretic_web_api import register_heretic_routes

app = Flask(__name__)
CORS(app)

# ... existing routes ...

# Register Heretic routes
register_heretic_routes(app)
```

### Step 2: Add Panel to Dashboard

Include the Heretic panel in your main dashboard template:

```html
<!-- In your main dashboard HTML -->
<div class="dashboard-panels">
    <!-- Existing panels -->
    <div class="panel">...</div>

    <!-- Add Heretic panel -->
    {% include 'heretic_panel.html' %}
</div>
```

Or load dynamically via JavaScript:

```javascript
// Load Heretic panel
fetch('/static/heretic_panel.html')
    .then(response => response.text())
    .then(html => {
        document.getElementById('hereticPanelContainer').innerHTML = html;
    });
```

### Step 3: Add Menu Link

Add Heretic to your navigation:

```html
<nav>
    <a href="#dashboard">Dashboard</a>
    <a href="#benchmark">Benchmark</a>
    <a href="#heretic" onclick="showPanel('hereticPanel')">🔬 Heretic</a>
</nav>
```

### What You Get

#### Full Web Interface
- **Configuration Tab** - Adjust optimization parameters
- **Abliterate Tab** - Run abliteration workflows
- **Evaluate Tab** - Test model safety
- **Models Tab** - Manage abliterated models
- **Datasets Tab** - View available datasets

#### REST API Endpoints

All available at `/api/heretic/*`:

```bash
# Get configuration
GET /api/heretic/config

# Update configuration
POST /api/heretic/config
  {
    "n_trials": 200,
    "max_batch_size": 128
  }

# List datasets
GET /api/heretic/datasets

# Start abliteration (async)
POST /api/heretic/abliterate
  {
    "model": "uncensored_code",
    "trials": 200,
    "save": true
  }
  → Returns: {"job_id": "abliterate_1"}

# Check job status
GET /api/heretic/abliterate/abliterate_1

# Evaluate model
POST /api/heretic/evaluate
  {
    "model": "uncensored_code"
  }

# List abliterated models
GET /api/heretic/models

# Get system status
GET /api/heretic/status
```

---

## RECOMMENDED: USE BOTH TOGETHER

```python
def initialize_components():
    """Initialize AI components"""

    # 1. Register Heretic hooks (automatic monitoring)
    from heretic_hook import register_heretic_hooks
    from hook_system import create_default_hooks

    hook_manager = create_default_hooks()
    register_heretic_hooks(
        hook_manager,
        enable_refusal_detection=True,
        enable_safety_monitor=True,
        enable_abliteration_trigger=True
    )

    # 2. Register Web API (manual control)
    from heretic_web_api import register_heretic_routes
    register_heretic_routes(app)

    print("✓ Heretic fully integrated (hooks + web API)")
```

**Benefits:**
- **Hooks** provide automatic monitoring in background
- **Web UI** provides manual control and visualization
- **Best of both worlds!**

---

## TESTING THE INTEGRATION

### Test Hook System

```python
# In Python console or test script
from hook_system import HookManager, HookContext, HookType
from heretic_hook import register_heretic_hooks
import time

manager = HookManager()
register_heretic_hooks(manager)

# Test refusal detection
test_context = HookContext(
    hook_type=HookType.POST_QUERY,
    timestamp=time.time(),
    data={"response": "I'm sorry, I can't help with that."},
    metadata={}
)

results = manager.execute_hooks(HookType.POST_QUERY, test_context)
print(f"Executed {len(results)} hooks")
for result in results:
    print(f"  - {result.message}")
```

### Test Web API

```bash
# Test status endpoint
curl http://localhost:5050/api/heretic/status

# Test configuration
curl http://localhost:5050/api/heretic/config

# Test datasets
curl http://localhost:5050/api/heretic/datasets

# Test abliteration (starts async job)
curl -X POST http://localhost:5050/api/heretic/abliterate \
  -H "Content-Type: application/json" \
  -d '{"model": "uncensored_code", "trials": 50}'
```

---

## CONFIGURATION VIA WEB INTERFACE

Once integrated, users can configure Heretic via TEMPEST dashboard:

### Configuration Panel
```
┌─────────────────────────────────────────┐
│ 🔬 HERETIC Configuration                │
├─────────────────────────────────────────┤
│ Number of Trials: [200    ] ↕          │
│ Startup Trials:   [60     ] ↕          │
│ Max Batch Size:   [128    ] ↕          │
│ KL Divergence:    [1.0    ] ↕          │
│                                         │
│ [💾 Save Configuration] [🔄 Reload]     │
└─────────────────────────────────────────┘
```

### Abliteration Workflow
```
┌─────────────────────────────────────────┐
│ Model: [uncensored_code ▼]             │
│ Trials: [200 ↕]                        │
│ ☑ Save abliterated model               │
│                                         │
│ [🚀 Start Abliteration]                 │
├─────────────────────────────────────────┤
│ Progress: ████████░░ 80%                │
│ Status: Running trial 160/200          │
└─────────────────────────────────────────┘
```

### Safety Metrics Dashboard
```
┌──────────────┬──────────────┬──────────────┐
│ Refusal Count│ Refusal Rate │ KL Divergence│
│      15      │    15.0%     │    0.23      │
└──────────────┴──────────────┴──────────────┘
```

---

## SECURITY CONSIDERATIONS

### Access Control
Add authentication to Heretic endpoints:

```python
from flask import request, abort

@heretic_bp.before_request
def check_auth():
    """Require authentication for Heretic operations"""
    if not security.is_authenticated(request):
        abort(403, "Unauthorized access to Heretic")
```

### Abliteration Approval
Require manual approval before abliterating:

```python
register_heretic_hooks(
    hook_manager,
    enable_abliteration_trigger=True,
    auto_abliterate=False,  # NEVER auto-abliterate
    refusal_threshold=0.5
)
```

### Audit Logging
Log all abliteration operations:

```python
# In heretic_web_api.py
@heretic_bp.route('/abliterate', methods=['POST'])
def abliterate_model():
    # Log operation
    audit_log.log(
        action="heretic_abliterate_start",
        user=request.user,
        model=model_name,
        timestamp=datetime.now()
    )
```

---

## FILE LOCATIONS

```
02-ai-engine/
├── heretic_hook.py              # Hook system integration ⭐
├── heretic_web_api.py           # Flask REST API routes ⭐
├── templates/
│   └── heretic_panel.html       # Web UI panel ⭐
├── heretic_config.toml          # Configuration file
├── heretic_config.py            # Config loader
├── heretic_abliteration.py      # Core engine
├── heretic_optimizer.py         # Optuna optimizer
├── heretic_evaluator.py         # Evaluator & detector
└── heretic_datasets.py          # Dataset management
```

---

## DEPENDENCIES

Already available in your environment:
- Flask (for web API)
- hook_system.py (for hooks)
- enhanced_ai_engine.py (for abliteration)

Install if needed:
```bash
pip install optuna pydantic pydantic-settings toml
```

---

## TROUBLESHOOTING

### Hooks Not Working
```python
# Check if hooks are registered
print(hook_manager.hooks)

# Check if Heretic modules available
from heretic_hook import HERETIC_AVAILABLE, HOOK_SYSTEM_AVAILABLE
print(f"Heretic: {HERETIC_AVAILABLE}, Hooks: {HOOK_SYSTEM_AVAILABLE}")
```

### Web API Not Available
```bash
# Check if routes registered
curl http://localhost:5050/api/heretic/status

# Check Flask logs
# Should see: "✓ Heretic Web API routes registered"
```

### Configuration Not Loading
```bash
# Check if config file exists
ls -la 02-ai-engine/heretic_config.toml

# Test config loading
python3 -c "from heretic_config import ConfigLoader; print(ConfigLoader.load())"
```

---

## NEXT STEPS

1. **Choose integration method** (hooks, web API, or both)
2. **Add integration code** to `ai_gui_dashboard.py`
3. **Test endpoints** with curl or browser
4. **Configure via TEMPEST** dashboard
5. **Run abliteration** workflow
6. **Monitor safety metrics** automatically

---

## SUPPORT

- **Hook System Docs:** See `hook_system.py` docstrings
- **Web API Reference:** See `heretic_web_api.py` endpoint docs
- **Core Modules:** See `HERETIC_INTEGRATION_PLAN.md`
- **Technical Details:** See `HERETIC_TECHNICAL_REPORT.md`

---

**INTEGRATION COMPLETE!** Heretic is now available via your DSMIL TEMPEST web interface with both automatic monitoring (hooks) and manual control (web API).
