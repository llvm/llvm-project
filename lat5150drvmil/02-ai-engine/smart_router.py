#!/usr/bin/env python3
"""
Smart Router - Intelligent Query Routing System

Automatically routes queries to the best model:
- Code tasks → DeepSeek Coder or Qwen Coder
- General queries → DeepSeek R1
- Complex analysis → CodeLlama 70B
- Multimodal → Gemini Pro
- Web search needed → Add web search first
- Adaptive Compute Scaling → Allocate budget based on difficulty (NEW)

100% automatic, invisible to user. Just type and get optimal results.
Version: 2.0.0 (with adaptive compute)
"""

import re
from typing import Tuple, Optional

# Adaptive compute scaling
try:
    from adaptive_compute import BudgetAllocator, DifficultyLevel
    ADAPTIVE_COMPUTE_AVAILABLE = True
except ImportError:
    ADAPTIVE_COMPUTE_AVAILABLE = False

class SmartRouter:
    def __init__(self, enable_adaptive_compute: bool = True):
        """
        Initialize smart router with detection patterns

        Args:
            enable_adaptive_compute: Enable test-time compute scaling (default: True)
        """

        # Code detection patterns (comprehensive)
        self.code_keywords = {
            'actions': ['write', 'create', 'implement', 'build', 'make', 'generate', 'develop', 'code'],
            'artifacts': ['function', 'class', 'method', 'module', 'script', 'program', 'api', 'endpoint', 'service'],
            'languages': ['python', 'javascript', 'typescript', 'java', 'c++', 'rust', 'go', 'php', 'ruby', 'swift'],
            'operations': ['debug', 'fix', 'refactor', 'optimize', 'test', 'deploy', 'compile'],
            'concepts': ['algorithm', 'data structure', 'regex', 'sql', 'http', 'rest', 'graphql', 'async']
        }

        # Web search detection patterns
        self.web_search_keywords = {
            'temporal': ['latest', 'recent', 'today', 'this week', 'current', 'now', 'update'],
            'questions': ['what happened', 'who is', 'when did', 'where is', 'news about'],
            'research': ['papers on', 'research about', 'find information', 'search for']
        }

        # NotebookLM detection patterns
        self.notebooklm_keywords = {
            'actions': ['summarize', 'create faq', 'study guide', 'synthesize', 'briefing', 'analyze documents'],
            'artifacts': ['sources', 'documents', 'papers', 'notes', 'research materials'],
            'operations': ['add source', 'create notebook', 'query sources', 'compare sources'],
            'modes': ['notebooklm', 'notebook', 'document analysis', 'research assistant']
        }

        # Geospatial analytics detection patterns
        self.geospatial_keywords = {
            'actions': ['map', 'visualize geo', 'plot location', 'show on map', 'geospatial analysis'],
            'artifacts': ['threat intel', 'osint', 'infrastructure', 'geo data', 'coordinates', 'locations'],
            'formats': ['kml', 'geojson', 'shapefile', 'gpx'],
            'operations': ['load geo', 'create map', 'threat mapping', 'infrastructure mapping'],
            'modes': ['geospatial', 'mapping', 'osint mapping', 'threat intel map']
        }

        # RDKit cheminformatics detection patterns
        self.rdkit_keywords = {
            'actions': ['analyze molecule', 'calculate descriptor', 'drug likeness', 'similarity search'],
            'artifacts': ['molecule', 'compound', 'smiles', 'structure', 'chemical', 'drug'],
            'operations': ['parse smiles', 'fingerprint', 'substructure', 'lipinski', 'molecular weight'],
            'modes': ['cheminformatics', 'rdkit', 'drug discovery', 'molecular analysis']
        }

        # PRT visualization detection patterns
        self.prt_keywords = {
            'actions': ['visualize data', 'pattern recognition', 'classify', 'cluster', 'train model'],
            'artifacts': ['dataset', 'features', 'training data', 'ml model', 'classifier'],
            'operations': ['load dataset', 'explore data', 'dimensionality reduction', 'pca', 'tsne'],
            'modes': ['data viz', 'pattern recognition', 'ml visualization', 'interactive viz']
        }

        # MxGPU virtualization detection patterns
        self.mxgpu_keywords = {
            'actions': ['setup gpu', 'configure sriov', 'passthrough', 'vm gpu', 'enable vf'],
            'artifacts': ['virtual machine', 'gpu', 'vf', 'pci device', 'hypervisor'],
            'operations': ['detect gpu', 'sriov status', 'iommu', 'vfio', 'kvm', 'xen'],
            'modes': ['mxgpu', 'gpu virtualization', 'sriov', 'gpu passthrough']
        }

        # NMDA agonist antidepressant analysis detection patterns
        self.nmda_keywords = {
            'actions': ['analyze nmda', 'nmda activity', 'antidepressant', 'bbb penetration', 'blood brain barrier'],
            'artifacts': ['nmda agonist', 'nmda antagonist', 'ketamine', 'esketamine', 'memantine', 'antidepressant'],
            'operations': ['nmda receptor', 'glutamate', 'rapid acting antidepressant', 'treatment resistant depression'],
            'modes': ['nmda analysis', 'antidepressant screening', 'glutamatergic', 'ketamine analog']
        }

        # NPS abuse potential analysis detection patterns
        self.nps_keywords = {
            'actions': ['abuse potential', 'recreational potential', 'nps', 'novel psychoactive', 'drug scheduling'],
            'artifacts': ['designer drug', 'synthetic cannabinoid', 'cathinone', 'fentanyl analog', 'research chemical'],
            'operations': ['classify nps', 'predict abuse', 'receptor binding', 'neurotoxicity', 'antidote'],
            'modes': ['nps analysis', 'abuse screening', 'drug threat intelligence', 'proactive drug policy', 'dark web']
        }

        # Pharmaceutical research corpus detection patterns (comprehensive)
        self.pharmaceutical_keywords = {
            'actions': ['drug discovery', 'admet', 'molecular docking', 'patient simulation', 'safety profile'],
            'artifacts': ['pharmaceutical', 'therapeutic', 'drug candidate', 'compound library', 'protocol'],
            'operations': ['screen compound', 'docking', 'bbb penetration', 'toxicity', 'pharmacokinetics'],
            'zeropain': ['zeropain', 'opioid protocol', 'pain management', 'analgesic', 'tolerance protection'],
            'capabilities': ['intel ai', 'autodock', 'virtual screening', 'pk/pd', 'regulatory dossier'],
            'modes': ['pharmaceutical corpus', 'drug development', 'therapeutic research', 'clinical optimization']
        }

        # Complexity indicators
        self.complex_indicators = ['comprehensive', 'detailed analysis', 'research', 'investigate', 'explore', 'compare multiple']

        # Initialize adaptive compute allocator
        if enable_adaptive_compute and ADAPTIVE_COMPUTE_AVAILABLE:
            self.budget_allocator = BudgetAllocator()
            print("✓ Adaptive compute scaling enabled")
        else:
            self.budget_allocator = None

    def detect_code_task(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if query is code-related

        Returns:
            (is_code, task_type, complexity)
            task_type: 'function', 'class', 'script', 'refactor', 'debug', 'explain', None
            complexity: 'simple', 'medium', 'complex', None
        """
        query_lower = query.lower()

        # Check for code keywords
        action_match = any(keyword in query_lower for keyword in self.code_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.code_keywords['artifacts'])
        language_match = any(keyword in query_lower for keyword in self.code_keywords['languages'])
        operation_match = any(keyword in query_lower for keyword in self.code_keywords['operations'])
        concept_match = any(keyword in query_lower for keyword in self.code_keywords['concepts'])

        # Code if: (action + artifact) OR language OR (operation + programming context)
        is_code = (action_match and artifact_match) or language_match or operation_match or concept_match

        if not is_code:
            return False, None, None

        # Determine task type
        task_type = 'general_code'
        if any(word in query_lower for word in ['function', 'method', 'def ']):
            task_type = 'function'
        elif any(word in query_lower for word in ['class', 'object', 'struct']):
            task_type = 'class'
        elif any(word in query_lower for word in ['script', 'program', 'tool']):
            task_type = 'script'
        elif any(word in query_lower for word in ['refactor', 'improve', 'optimize']):
            task_type = 'refactor'
        elif any(word in query_lower for word in ['debug', 'fix', 'bug', 'error']):
            task_type = 'debug'
        elif any(word in query_lower for word in ['explain', 'what does', 'how does']):
            task_type = 'explain'

        # Determine complexity
        complexity = 'medium'  # default
        if any(word in query_lower for word in ['simple', 'basic', 'quick', 'small', 'snippet']):
            complexity = 'simple'
        elif any(word in query_lower for word in ['complex', 'system', 'architecture', 'framework', 'large']):
            complexity = 'complex'

        # Override: explain is always simple
        if task_type == 'explain':
            complexity = 'simple'

        return True, task_type, complexity

    def detect_web_search_needed(self, query: str) -> bool:
        """Detect if query needs web search"""
        query_lower = query.lower()

        # Check for temporal or research keywords
        temporal_match = any(keyword in query_lower for keyword in self.web_search_keywords['temporal'])
        question_match = any(keyword in query_lower for keyword in self.web_search_keywords['questions'])
        research_match = any(keyword in query_lower for keyword in self.web_search_keywords['research'])

        return temporal_match or question_match or research_match

    def detect_notebooklm_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is NotebookLM-related

        Returns:
            (is_notebooklm, task_mode)
            task_mode: 'summarize', 'faq', 'study_guide', 'synthesis', 'briefing', 'qa', None
        """
        query_lower = query.lower()

        # Check for NotebookLM keywords
        action_match = any(keyword in query_lower for keyword in self.notebooklm_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.notebooklm_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.notebooklm_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.notebooklm_keywords['modes'])

        is_notebooklm = action_match or (artifact_match and operation_match) or mode_match

        if not is_notebooklm:
            return False, None

        # Determine task mode
        task_mode = 'qa'  # default
        if 'summarize' in query_lower or 'summary' in query_lower:
            task_mode = 'summarize'
        elif 'faq' in query_lower or 'frequently asked' in query_lower:
            task_mode = 'faq'
        elif 'study guide' in query_lower or 'study material' in query_lower:
            task_mode = 'study_guide'
        elif 'synthesize' in query_lower or 'synthesis' in query_lower or 'compare sources' in query_lower:
            task_mode = 'synthesis'
        elif 'briefing' in query_lower or 'executive summary' in query_lower:
            task_mode = 'briefing'

        return True, task_mode

    def detect_geospatial_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is geospatial-related

        Returns:
            (is_geospatial, task_mode)
            task_mode: 'load_data', 'create_map', 'threat_intel', 'osint', None
        """
        query_lower = query.lower()

        # Check for geospatial keywords
        action_match = any(keyword in query_lower for keyword in self.geospatial_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.geospatial_keywords['artifacts'])
        format_match = any(keyword in query_lower for keyword in self.geospatial_keywords['formats'])
        operation_match = any(keyword in query_lower for keyword in self.geospatial_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.geospatial_keywords['modes'])

        is_geospatial = action_match or (artifact_match and operation_match) or mode_match or format_match

        if not is_geospatial:
            return False, None

        # Determine task mode
        task_mode = 'create_map'  # default
        if 'load' in query_lower or 'import' in query_lower:
            task_mode = 'load_data'
        elif 'threat' in query_lower or 'osint' in query_lower:
            task_mode = 'threat_intel'
        elif 'infrastructure' in query_lower:
            task_mode = 'infrastructure'

        return True, task_mode

    def detect_rdkit_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is cheminformatics-related

        Returns:
            (is_rdkit, task_mode)
            task_mode: 'parse', 'descriptors', 'fingerprint', 'similarity', 'drug_likeness', None
        """
        query_lower = query.lower()

        # Check for RDKit keywords
        action_match = any(keyword in query_lower for keyword in self.rdkit_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.rdkit_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.rdkit_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.rdkit_keywords['modes'])

        is_rdkit = action_match or artifact_match or operation_match or mode_match

        if not is_rdkit:
            return False, None

        # Determine task mode
        task_mode = 'parse'  # default
        if 'descriptor' in query_lower or 'property' in query_lower or 'molecular weight' in query_lower:
            task_mode = 'descriptors'
        elif 'fingerprint' in query_lower:
            task_mode = 'fingerprint'
        elif 'similarity' in query_lower or 'similar' in query_lower:
            task_mode = 'similarity'
        elif 'drug' in query_lower or 'lipinski' in query_lower:
            task_mode = 'drug_likeness'
        elif 'substructure' in query_lower:
            task_mode = 'substructure'

        return True, task_mode

    def detect_prt_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is data visualization/ML-related

        Returns:
            (is_prt, task_mode)
            task_mode: 'load', 'visualize', 'classify', 'cluster', 'reduce', None
        """
        query_lower = query.lower()

        # Check for PRT keywords
        action_match = any(keyword in query_lower for keyword in self.prt_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.prt_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.prt_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.prt_keywords['modes'])

        is_prt = action_match or (artifact_match and operation_match) or mode_match

        if not is_prt:
            return False, None

        # Determine task mode
        task_mode = 'visualize'  # default
        if 'load' in query_lower or 'import' in query_lower:
            task_mode = 'load'
        elif 'classify' in query_lower or 'train' in query_lower:
            task_mode = 'classify'
        elif 'cluster' in query_lower:
            task_mode = 'cluster'
        elif 'pca' in query_lower or 'tsne' in query_lower or 'dimensionality' in query_lower:
            task_mode = 'reduce'
        elif 'explore' in query_lower:
            task_mode = 'explore'

        return True, task_mode

    def detect_mxgpu_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is GPU virtualization-related

        Returns:
            (is_mxgpu, task_mode)
            task_mode: 'detect', 'sriov', 'passthrough', 'config', None
        """
        query_lower = query.lower()

        # Check for MxGPU keywords
        action_match = any(keyword in query_lower for keyword in self.mxgpu_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.mxgpu_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.mxgpu_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.mxgpu_keywords['modes'])

        is_mxgpu = action_match or (artifact_match and operation_match) or mode_match

        if not is_mxgpu:
            return False, None

        # Determine task mode
        task_mode = 'detect'  # default
        if 'sriov' in query_lower or 'virtual function' in query_lower:
            task_mode = 'sriov'
        elif 'passthrough' in query_lower or 'iommu' in query_lower:
            task_mode = 'passthrough'
        elif 'config' in query_lower or 'setup' in query_lower or 'vm' in query_lower:
            task_mode = 'config'
        elif 'status' in query_lower or 'check' in query_lower:
            task_mode = 'status'

        return True, task_mode

    def detect_nmda_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is NMDA agonist antidepressant analysis-related

        Returns:
            (is_nmda, task_mode)
            task_mode: 'nmda_activity', 'bbb_prediction', 'compare', 'comprehensive', None
        """
        query_lower = query.lower()

        # Check for NMDA keywords
        action_match = any(keyword in query_lower for keyword in self.nmda_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.nmda_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.nmda_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.nmda_keywords['modes'])

        is_nmda = action_match or artifact_match or operation_match or mode_match

        if not is_nmda:
            return False, None

        # Determine task mode
        task_mode = 'nmda_activity'  # default
        if 'bbb' in query_lower or 'blood brain barrier' in query_lower:
            task_mode = 'bbb_prediction'
        elif 'compare' in query_lower or 'similarity' in query_lower:
            task_mode = 'compare'
        elif 'comprehensive' in query_lower or 'full analysis' in query_lower:
            task_mode = 'comprehensive'

        return True, task_mode

    def detect_nps_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is NPS abuse potential analysis-related

        Returns:
            (is_nps, task_mode)
            task_mode: 'classify', 'abuse_potential', 'receptor_binding', 'antidote', 'batch', None
        """
        query_lower = query.lower()

        # Check for NPS keywords
        action_match = any(keyword in query_lower for keyword in self.nps_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.nps_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.nps_keywords['operations'])
        mode_match = any(keyword in query_lower for keyword in self.nps_keywords['modes'])

        is_nps = action_match or artifact_match or operation_match or mode_match

        if not is_nps:
            return False, None

        # Determine task mode
        task_mode = 'abuse_potential'  # default
        if 'classify' in query_lower or 'classification' in query_lower:
            task_mode = 'classify'
        elif 'receptor' in query_lower or 'binding' in query_lower:
            task_mode = 'receptor_binding'
        elif 'antidote' in query_lower or 'overdose' in query_lower:
            task_mode = 'antidote'
        elif 'batch' in query_lower or 'screen' in query_lower:
            task_mode = 'batch'

        return True, task_mode

    def detect_pharmaceutical_task(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is pharmaceutical research corpus-related

        Returns:
            (is_pharmaceutical, task_mode)
            task_mode: 'screen', 'docking', 'admet', 'safety', 'simulation', 'zeropain', None
        """
        query_lower = query.lower()

        # Check for pharmaceutical keywords
        action_match = any(keyword in query_lower for keyword in self.pharmaceutical_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.pharmaceutical_keywords['artifacts'])
        operation_match = any(keyword in query_lower for keyword in self.pharmaceutical_keywords['operations'])
        zeropain_match = any(keyword in query_lower for keyword in self.pharmaceutical_keywords['zeropain'])
        capability_match = any(keyword in query_lower for keyword in self.pharmaceutical_keywords['capabilities'])
        mode_match = any(keyword in query_lower for keyword in self.pharmaceutical_keywords['modes'])

        # Also check if this is NMDA or NPS task (they're part of pharmaceutical corpus)
        is_nmda, _ = self.detect_nmda_task(query)
        is_nps, _ = self.detect_nps_task(query)

        is_pharmaceutical = (action_match or artifact_match or operation_match or
                           zeropain_match or capability_match or mode_match or
                           is_nmda or is_nps)

        if not is_pharmaceutical:
            return False, None

        # Determine task mode
        task_mode = 'screen'  # default
        if 'docking' in query_lower or 'dock' in query_lower or 'binding affinity' in query_lower:
            task_mode = 'docking'
        elif 'admet' in query_lower or 'pharmacokinetics' in query_lower or 'pk/pd' in query_lower:
            task_mode = 'admet'
        elif 'safety' in query_lower or 'toxicity' in query_lower or 'adverse' in query_lower:
            task_mode = 'safety'
        elif 'patient simulation' in query_lower or 'clinical trial' in query_lower or 'protocol' in query_lower:
            task_mode = 'simulation'
        elif zeropain_match or 'pain management' in query_lower or 'opioid' in query_lower:
            task_mode = 'zeropain'
        elif is_nmda:
            task_mode = 'nmda_antidepressant'
        elif is_nps:
            task_mode = 'nps_abuse'

        return True, task_mode

    def route(self, query: str, has_images: bool = False, user_preference: str = None) -> dict:
        """
        Main routing function - decides which model to use with adaptive compute

        Args:
            query: User's query
            has_images: Whether query includes images
            user_preference: Optional override ("code", "fast", "large", etc.)

        Returns:
            Dict with routing decision including compute budget
        """

        # User override takes precedence
        if user_preference and user_preference != "auto":
            return {
                "model": user_preference,
                "reason": "user_preference",
                "explanation": f"User selected {user_preference}"
            }

        # Multimodal detection
        if has_images:
            return {
                "model": "gemini-pro",
                "reason": "multimodal",
                "explanation": "Images detected - Gemini required",
                "web_search": False
            }

        # NotebookLM detection
        is_notebooklm, task_mode = self.detect_notebooklm_task(query)
        if is_notebooklm:
            return {
                "model": "notebooklm",
                "reason": "notebooklm_task",
                "task_mode": task_mode,
                "explanation": f"NotebookLM task detected: {task_mode}",
                "web_search": False
            }

        # Geospatial analytics detection
        is_geospatial, geo_mode = self.detect_geospatial_task(query)
        if is_geospatial:
            return {
                "model": "geospatial",
                "reason": "geospatial_task",
                "task_mode": geo_mode,
                "explanation": f"Geospatial analytics task: {geo_mode}",
                "web_search": False
            }

        # RDKit cheminformatics detection
        is_rdkit, rdkit_mode = self.detect_rdkit_task(query)
        if is_rdkit:
            return {
                "model": "rdkit",
                "reason": "rdkit_task",
                "task_mode": rdkit_mode,
                "explanation": f"Cheminformatics task: {rdkit_mode}",
                "web_search": False
            }

        # PRT visualization detection
        is_prt, prt_mode = self.detect_prt_task(query)
        if is_prt:
            return {
                "model": "prt",
                "reason": "prt_task",
                "task_mode": prt_mode,
                "explanation": f"Data visualization/ML task: {prt_mode}",
                "web_search": False
            }

        # Pharmaceutical research corpus detection (includes NMDA, NPS, ZEROPAIN)
        is_pharmaceutical, pharma_mode = self.detect_pharmaceutical_task(query)
        if is_pharmaceutical:
            return {
                "model": "pharmaceutical",
                "reason": "pharmaceutical_task",
                "task_mode": pharma_mode,
                "explanation": f"Pharmaceutical research corpus: {pharma_mode}",
                "web_search": False
            }

        # NMDA agonist antidepressant analysis detection (fallback if not caught by pharmaceutical)
        is_nmda, nmda_mode = self.detect_nmda_task(query)
        if is_nmda:
            return {
                "model": "nmda",
                "reason": "nmda_task",
                "task_mode": nmda_mode,
                "explanation": f"NMDA antidepressant analysis: {nmda_mode}",
                "web_search": False
            }

        # NPS abuse potential analysis detection (fallback if not caught by pharmaceutical)
        is_nps, nps_mode = self.detect_nps_task(query)
        if is_nps:
            return {
                "model": "nps",
                "reason": "nps_task",
                "task_mode": nps_mode,
                "explanation": f"NPS abuse potential analysis: {nps_mode}",
                "web_search": False
            }

        # MxGPU virtualization detection
        is_mxgpu, mxgpu_mode = self.detect_mxgpu_task(query)
        if is_mxgpu:
            return {
                "model": "mxgpu",
                "reason": "mxgpu_task",
                "task_mode": mxgpu_mode,
                "explanation": f"GPU virtualization task: {mxgpu_mode}",
                "web_search": False
            }

        # Web search detection
        needs_web = self.detect_web_search_needed(query)

        # Allocate compute budget if available
        compute_budget = None
        difficulty = None
        if self.budget_allocator:
            budget, diff, confidence = self.budget_allocator.allocate(query)
            compute_budget = budget.to_dict()
            difficulty = diff.value

        # Code detection
        is_code, task_type, complexity = self.detect_code_task(query)

        if is_code:
            # Use budget allocator model if available, otherwise use legacy routing
            if compute_budget:
                model_map = {
                    "fast": "deepseek-coder:6.7b-instruct",
                    "code": "deepseek-coder:6.7b-instruct",
                    "large": "qwen2.5-coder:7b"
                }
                model = model_map.get(compute_budget['model'], "deepseek-coder:6.7b-instruct")
                explanation = f"Code task: {task_type} ({difficulty} difficulty)"
            else:
                # Legacy routing
                if complexity == 'simple' or task_type == 'explain':
                    model = "deepseek-coder:6.7b-instruct"
                    explanation = f"Code task detected: {task_type} ({complexity})"
                elif complexity == 'complex':
                    model = "qwen2.5-coder:7b"
                    explanation = f"Complex code task: {task_type}"
                else:
                    model = "deepseek-coder:6.7b-instruct"
                    explanation = f"Code task: {task_type}"

            return {
                "model": model,
                "reason": "code_detected",
                "task_type": task_type,
                "complexity": complexity,
                "explanation": explanation,
                "web_search": needs_web,
                "compute_budget": compute_budget,
                "difficulty": difficulty
            }

        # General queries with adaptive compute
        if compute_budget:
            # Map budget model to actual model names
            model_map = {
                "fast": "deepseek-r1:1.5b",
                "code": "deepseek-coder:6.7b-instruct",
                "large": "codellama:70b"
            }
            model = model_map.get(compute_budget['model'], "deepseek-r1:1.5b")
            explanation = f"{difficulty.capitalize()} query - allocated {compute_budget['max_iterations']} iterations"

            return {
                "model": model,
                "reason": "adaptive_compute",
                "difficulty": difficulty,
                "explanation": explanation,
                "web_search": needs_web,
                "compute_budget": compute_budget
            }

        # Legacy routing (fallback if adaptive compute unavailable)
        query_length = len(query)
        word_count = len(query.split())

        # Check for complexity indicators
        is_complex = any(indicator in query.lower() for indicator in self.complex_indicators)

        if is_complex or query_length > 300 or word_count > 50:
            return {
                "model": "codellama:70b",
                "reason": "complex_query",
                "explanation": "Complex analysis detected",
                "web_search": needs_web
            }

        # Default: fast general model
        return {
            "model": "deepseek-r1:1.5b",
            "reason": "general_query",
            "explanation": "General question",
            "web_search": needs_web
        }

    def explain_routing(self, routing_decision: dict, format: str = "text") -> str:
        """
        Generate human-readable explanation of routing decision

        Args:
            routing_decision: Dict from route()
            format: "text", "emoji", "short"

        Returns:
            Formatted explanation string
        """
        model = routing_decision['model']
        reason = routing_decision.get('explanation', routing_decision['reason'])

        if format == "emoji":
            emoji_map = {
                'code_detected': '💻',
                'complex_query': '🧠',
                'general_query': '💬',
                'multimodal': '🖼️',
                'user_preference': '👤'
            }
            emoji = emoji_map.get(routing_decision['reason'], '🤖')
            return f"{emoji} {model.split(':')[0]} | {reason}"

        elif format == "short":
            model_short = model.split(':')[0].replace('-', ' ').title()
            return f"{model_short} ({reason})"

        else:  # text
            return f"Using {model} - {reason}"

# CLI
if __name__ == "__main__":
    import sys
    import json

    router = SmartRouter()

    if len(sys.argv) < 2:
        print("Smart Router - Usage:")
        print("  python3 smart_router.py 'write a python function'")
        print("  python3 smart_router.py 'what is quantum computing'")
        print("  python3 smart_router.py 'latest news about AI'")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    decision = router.route(query)

    print(json.dumps(decision, indent=2))
    print(f"\n{router.explain_routing(decision, format='emoji')}")
