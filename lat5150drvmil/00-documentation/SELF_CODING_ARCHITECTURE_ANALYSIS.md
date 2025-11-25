# DSMIL Self-Coding Architecture - Component Analysis

## Overview
This document maps the existing self-coding components in the LAT5150DRVMIL system to enable proper integration with file manager context menus and terminal APIs.

**Date**: 2025-11-20
**Purpose**: Blueprint for right-click "Open DSMIL AI" integration

---

## Existing Self-Coding Components

### 1. **Core Self-Improvement System**
**File**: `02-ai-engine/autonomous_self_improvement.py`

**Capabilities**:
- Performance monitoring (CPU, memory, latency, cache hits)
- Bottleneck detection and analysis
- Autonomous code modification proposals
- Meta-learning from interactions
- Emergence-friendly architecture

**Key Classes**:
- `PerformanceMetric`: Tracks performance measurements
- `ImprovementProposal`: Proposed system improvements
- `LearningInsight`: Learned patterns and optimizations
- `AutonomousSelfImprovement`: Main self-improvement engine

**Integration Points**:
- PostgreSQL for persistent improvement history
- File-based proposals and learning logs
- Can monitor and improve other code components

---

### 2. **Code Specialist**
**File**: `02-ai-engine/code_specialist.py`

**Capabilities**:
- Specialized coding models (DeepSeek Coder, Qwen 2.5 Coder, CodeLlama)
- Auto-routing based on task complexity
- Code generation, refactoring, debugging, review

**Key Models**:
```python
"fast_code": "deepseek-coder:6.7b-instruct"      # Quick snippets
"quality_code": "qwen2.5-coder:7b"               # Complex implementations
"review": "codellama:70b"                        # Code review
"large_review": "codellama:70b"                  # Deep analysis
```

**Task Detection**:
- Pattern-based classification (function, class, script, refactor, debug, review, explain)
- Complexity detection (simple, medium, complex)
- Automatic model selection

**Integration Points**:
- Uses `DSMILAIEngine` backend
- Can be invoked programmatically
- Supports streaming responses

---

### 3. **RAG-Enhanced Code Assistant**
**File**: `04-integrations/rag_system/code_assistant.py`

**Capabilities**:
- Multi-turn conversations with context
- Code execution and testing
- File operations (save/load/edit)
- Syntax highlighting
- Code review and analysis
- RAG integration with documentation
- 100% local (Ollama + transformers)

**Advanced Features**:
- Security vulnerability scanning (SQL injection, command injection, etc.)
- Performance optimization analysis
- AST-based code transformations
- Automatic documentation generation
- Unit test generation (pytest/unittest)
- Code complexity metrics
- Code smell detection

**Key Classes**:
- `Conversation`: Maintains conversation context
- `CodeAssistant`: Main assistant with RAG + local LLM

**Integration Points**:
- Uses Ollama for LLM inference
- Transformer-based RAG retriever
- Temporal awareness for documentation
- Can execute generated code safely

---

### 4. **98-Agent System**
**File**: `02-ai-engine/comprehensive_98_agent_system.py`

**Capabilities**:
- 98 specialized agents in 7 categories
- NPU/GNA heterogeneous execution
- Formal communication protocol
- Quality gates and dependencies
- AVX512 operations pinned to P-cores
- Graceful hardware fallback

**Agent Categories**:
1. **Strategic**: Planning, architecture, decisions
2. **Development**: Coding, implementation
3. **Infrastructure**: System setup, deployment
4. **Security**: Auditing, hardening
5. **QA**: Testing, validation
6. **Documentation**: Technical writing
7. **Operations**: Monitoring, maintenance

**Hardware Backends**:
- NPU (34-49.4 TOPS military mode)
- GNA (Gaussian Neural Accelerator)
- CPU P-cores (AVX512)
- CPU E-cores
- CPU Fallback

**Integration Points**:
- OpenVINO for NPU/GNA
- Agent messaging system
- Task orchestration
- Hardware capability detection

---

### 5. **CLI Interfaces**

#### DSMIL AI CLI
**File**: `02-ai-engine/dsmil_ai_cli.py`

**Commands**:
- `query <prompt>`: Simple AI query
- `reason <prompt>`: Deep reasoning
- `benchmark [tasks]`: Run benchmarks
- `security report`: Security status
- `stats`: System statistics
- `interactive`: Interactive mode
- `test`: System test

**Integration Points**:
- Uses `AISystemIntegrator`
- Security validation with `SecurityHardening`
- Model selection
- RAG and cache control

#### Enhanced AI CLI
**File**: `02-ai-engine/enhanced_ai_cli.py`
- Extended capabilities
- Additional model options
- Enhanced reasoning modes

---

### 6. **Supporting Components**

#### Neural Code Synthesis
**File**: `04-integrations/rag_system/neural_code_synthesis.py`
- Advanced code generation
- Neural architecture search
- Code completion and suggestions

#### Code Generators
**File**: `04-integrations/rag_system/code_generators.py`
- Documentation generator
- Test generator
- Boilerplate code generation

#### Code Transformers
**File**: `04-integrations/rag_system/code_transformers.py`
- Error handling transformations
- Type hint addition
- Performance refactoring

#### Code Validators
**File**: `04-integrations/rag_system/code_validator.py`
- Syntax validation
- Security checks
- Best practices enforcement

---

## Current Gaps

### What's Missing for Right-Click Integration:

1. **No File Manager Integration**
   - No existing `.desktop` files for context menus
   - No Nautilus/Thunar scripts
   - No file manager service files

2. **No Terminal API**
   - No standardized terminal interface
   - No IPC mechanism for external processes
   - No socket/pipe-based communication

3. **No Context-Aware Launcher**
   - No automatic working directory detection
   - No file type detection for specialized assistance
   - No project detection (git repos, package.json, etc.)

4. **No Session Management**
   - No persistent coding sessions per directory
   - No conversation history per project
   - No workspace state saving/loading

---

## Proposed Integration Architecture

### Components to Build:

```
┌─────────────────────────────────────────────────────────────────┐
│                 File Manager Integration                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Context Menu Script                                         │
│     - ~/.local/share/nautilus/scripts/Open DSMIL AI            │
│     - ~/.local/share/thunar/sendto/dsmil-ai.desktop           │
│                                                                 │
│  2. Terminal API Server                                         │
│     - Unix domain socket: /tmp/dsmil-ai.sock                   │
│     - JSON-RPC protocol                                         │
│     - Authentication and session management                     │
│                                                                 │
│  3. Context-Aware Launcher                                      │
│     - Detect project type (Python, C, Rust, etc.)              │
│     - Load project context (git, README, docs)                 │
│     - Initialize appropriate code specialist                    │
│                                                                 │
│  4. Session Manager                                             │
│     - Per-directory conversation history                        │
│     - Workspace state persistence                               │
│     - Recent files and changes tracking                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Flow:

```
User Right-Clicks Folder
        ↓
Context Menu Script Triggered
        ↓
Detect Project Context
  • Git repository?
  • Programming language?
  • Existing DSMIL session?
        ↓
Launch Terminal API Server (if not running)
        ↓
Connect to Server via Socket
        ↓
Load Project Context into RAG
  • README files
  • Source code structure
  • Documentation
  • Git history
        ↓
Initialize Code Assistant
  • Select appropriate model
  • Load conversation history
  • Set working directory
        ↓
Open Interactive Terminal Session
  • Code generation
  • File operations
  • Testing and debugging
  • Documentation
```

---

## Integration Hooks

### 1. **File Manager Context Menu**
```bash
# Nautilus script location
~/.local/share/nautilus/scripts/Open DSMIL AI

# Thunar custom action
~/.local/share/Thunar/sendto/dsmil-ai.desktop
```

### 2. **Terminal API Entry Points**

```python
# From autonomous_self_improvement.py
class AutonomousSelfImprovement:
    def analyze_directory(self, path: Path) -> Dict
    def propose_improvements(self, codebase: Path) -> List[ImprovementProposal]
    def monitor_project(self, path: Path) -> PerformanceMetric

# From code_specialist.py
class CodeSpecialist:
    def detect_code_task(self, query: str) -> Tuple[bool, str, str]
    def generate_code(self, prompt: str, context: str) -> str
    def review_code(self, code: str, filename: str) -> Dict

# From code_assistant.py
class CodeAssistant:
    def set_project_root(self, path: Path)
    def load_project_context(self) -> List[Dict]
    def generate_with_context(self, prompt: str) -> str
    def execute_code(self, code: str) -> Tuple[int, str, str]
```

### 3. **Session Management Hooks**

```python
class DSMILCodingSession:
    def create_session(self, working_dir: Path) -> str  # Returns session_id
    def restore_session(self, session_id: str) -> bool
    def save_conversation(self, session_id: str) -> bool
    def get_session_history(self, session_id: str) -> List[Dict]
```

### 4. **Project Context Hooks**

```python
class ProjectAnalyzer:
    def detect_project_type(self, path: Path) -> str  # python, c, rust, etc.
    def find_entry_points(self, path: Path) -> List[Path]
    def extract_dependencies(self, path: Path) -> Dict
    def analyze_structure(self, path: Path) -> Dict
```

---

## File Manager Support Matrix

| File Manager | Integration Method | Status |
|--------------|-------------------|---------|
| Nautilus (GNOME) | Scripts in `~/.local/share/nautilus/scripts/` | ✅ Supported |
| Thunar (XFCE) | Custom actions in `~/.local/share/Thunar/sendto/` | ✅ Supported |
| Dolphin (KDE) | Service menus in `~/.local/share/kservices5/ServiceMenus/` | ✅ Supported |
| Nemo (Cinnamon) | Scripts in `~/.local/share/nemo/scripts/` | ✅ Supported |
| Caja (MATE) | Scripts in `~/.local/share/caja/scripts/` | ✅ Supported |

---

## Next Steps

### Phase 1: Terminal API Server
1. Create Unix domain socket server
2. Implement JSON-RPC protocol
3. Add authentication and session management
4. Integrate with existing code components

### Phase 2: Context Menu Integration
1. Create Nautilus script
2. Create Thunar custom action
3. Create .desktop files for other file managers
4. Add project context detection

### Phase 3: Session Management
1. Implement per-directory sessions
2. Add conversation history persistence
3. Create workspace state management
4. Add recent files tracking

### Phase 4: Enhanced Features
1. Multi-project workspace support
2. Real-time file watching and suggestions
3. Integrated terminal with AI assistance
4. Visual Studio Code extension compatibility

---

## Security Considerations

1. **Socket Permissions**: Unix domain socket should be user-only (0600)
2. **Authentication**: Token-based auth for API access
3. **Sandboxing**: Code execution should be sandboxed
4. **Input Validation**: All inputs must be validated (already implemented in `security_hardening.py`)
5. **File Access Control**: Restrict access to user's projects only

---

## Performance Considerations

1. **Lazy Loading**: Don't load all project context upfront
2. **Caching**: Cache parsed ASTs and analyzed files
3. **Hardware Acceleration**: Use NPU/GNA when available
4. **Background Processing**: Run heavy analysis in background threads
5. **Session Persistence**: Save sessions asynchronously

---

## Conclusion

The LAT5150DRVMIL system has a comprehensive self-coding architecture with:
- ✅ Advanced code generation (multiple specialized models)
- ✅ Code analysis and review capabilities
- ✅ RAG-enhanced context awareness
- ✅ Self-improvement and meta-learning
- ✅ 98-agent system for complex tasks

**Missing components** for right-click integration:
- ❌ File manager context menu scripts
- ❌ Terminal API server with IPC
- ❌ Context-aware session manager
- ❌ Project detection and initialization

Next commit will implement these missing components for seamless "Open DSMIL AI" functionality.
