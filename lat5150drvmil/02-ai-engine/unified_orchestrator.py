#!/usr/bin/env python3
"""
DSMIL Unified AI Orchestrator - LOCAL-FIRST Architecture

Philosophy:
1. DEFAULT: Local DeepSeek (privacy, no guardrails, zero cost, DSMIL-attested)
2. Gemini: ONLY for multimodal (images/video local can't handle)
3. OpenAI: ONLY when explicitly requested by user
4. All cloud backends OPTIONAL - graceful degradation to local

Routing Priority:
  Multimodal query → Try Gemini → Fallback to local
  Explicit request → Use requested backend → Fallback to local
  Everything else → Local DeepSeek (default)
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE
from sub_agents.gemini_wrapper import GeminiAgent
from sub_agents.openai_wrapper import OpenAIAgent
from sub_agents.notebooklm_wrapper import NotebookLMAgent
from sub_agents.geospatial_wrapper import GeospatialAgent
from sub_agents.rdkit_wrapper import RDKitAgent
from sub_agents.prt_visualization_wrapper import PRTVisualizationAgent
from sub_agents.mxgpu_wrapper import MxGPUAgent
from sub_agents.nmda_agonist_analyzer import NMDAAgonistAnalyzer
from sub_agents.nps_abuse_potential_analyzer import NPSAbusePotentialAnalyzer
from sub_agents.pharmaceutical_corpus import PharmaceuticalCorpus
from sub_agents.pharmaceutical_conversation_agent import PharmaceuticalConversationAgent
from smart_router import SmartRouter
from web_search import WebSearch
from shodan_search import ShodanSearch

# Pydantic models for type-safe orchestration
if PYDANTIC_AVAILABLE:
    from pydantic_models import (
        OrchestratorResponse,
        OrchestratorRequest,
        RoutingDecision,
        BackendType,
        RoutingReason,
        WebSearchMeta,
        WebSearchResult,
        ShodanSearchMeta,
    )

# ACE-FCA (Advanced Context Engineering for Coding Agents)
try:
    from ace_context_engine import ACEContextEngine, PhaseType
    from ace_workflow_orchestrator import ACEWorkflowOrchestrator, WorkflowTask
    from ace_subagents import create_subagent
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False

# HumanLayer-inspired enhancements
try:
    from parallel_agent_executor import ParallelAgentExecutor
    from worktree_manager import WorktreeManager
    from task_distributor import TaskDistributor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

class UnifiedAIOrchestrator:
    def __init__(self, enable_ace=True, pydantic_mode=False):
        """
        Initialize unified orchestrator with optional Pydantic support

        Args:
            enable_ace: Enable ACE-FCA context engineering
            pydantic_mode: If True, return Pydantic models instead of dicts
        """
        self.pydantic_mode = pydantic_mode and PYDANTIC_AVAILABLE

        # Primary: Local AI (always available)
        self.local = DSMILAIEngine(pydantic_mode=pydantic_mode)

        # Smart Router (NEW - automatic query routing)
        self.router = SmartRouter()

        # Web Search (NEW - for current information)
        self.web = WebSearch()

        # Shodan Search (NEW - for cybersecurity/threat intelligence)
        self.shodan = ShodanSearch()

        # Optional: Cloud backends (graceful degradation)
        self.gemini = GeminiAgent()
        self.openai = OpenAIAgent()
        self.notebooklm = NotebookLMAgent()

        # Specialized agents (local)
        self.geospatial = GeospatialAgent()
        self.rdkit = RDKitAgent()
        self.prt = PRTVisualizationAgent()
        self.mxgpu = MxGPUAgent()
        self.nmda = NMDAAgonistAnalyzer()
        self.nps = NPSAbusePotentialAnalyzer(verbose=False)
        self.pharmaceutical = PharmaceuticalCorpus(tempest_level=1, user_id="orchestrator")  # Default TEMPEST Level 1
        self.pharma_conversation = PharmaceuticalConversationAgent(tempest_level=1, user_id="orchestrator")  # Conversational AI

        # ACE-FCA (Advanced Context Engineering)
        self.ace_enabled = enable_ace and ACE_AVAILABLE
        if self.ace_enabled:
            self.ace_engine = ACEContextEngine(max_tokens=8192)
            self.ace_workflow = ACEWorkflowOrchestrator(
                ai_engine=self.local,
                enable_human_review=True
            )
        else:
            self.ace_engine = None
            self.ace_workflow = None

        # HumanLayer-inspired parallel execution
        self.parallel_enabled = PARALLEL_AVAILABLE
        if self.parallel_enabled:
            self.parallel_executor = ParallelAgentExecutor(
                orchestrator=self,
                max_concurrent_agents=3
            )
            self.worktree_manager = WorktreeManager()
            self.task_distributor = TaskDistributor()
        else:
            self.parallel_executor = None
            self.worktree_manager = None
            self.task_distributor = None

        print("🎯 DSMIL Unified Orchestrator - LOCAL-FIRST + SMART ROUTING + ACE-FCA + PARALLEL")
        print(f"   Local AI: ✅ DeepSeek R1 + DeepSeek Coder + Qwen Coder")
        print(f"   Smart Router: ✅ Auto-detects code/general/complex/specialized queries")
        print(f"   ACE-FCA: {'✅ Context Engineering Enabled' if self.ace_enabled else '⚠️  Not available'}")
        print(f"   Parallel Execution: {'✅ M U L T I C L A U D E' if self.parallel_enabled else '⚠️  Not available'}")
        print(f"   Gemini Pro: {'✅ Available' if self.gemini.is_available() else '⚠️  Not configured'} (multimodal only)")
        print(f"   NotebookLM: {'✅ Available' if self.notebooklm.is_available() else '⚠️  Not configured'} (document research)")
        print(f"   Geospatial: {'✅ Available' if self.geospatial.is_available() else '⚠️  Dependencies missing'} (OSINT, threat-intel mapping)")
        print(f"   RDKit: {'✅ Available' if self.rdkit.is_available() else '⚠️  Dependencies missing'} (cheminformatics, drug discovery)")
        print(f"   PRT Viz: {'✅ Available' if self.prt.is_available() else '⚠️  Dependencies missing'} (data viz, ML, pattern recognition)")
        print(f"   MxGPU: {'✅ Available' if self.mxgpu.is_available() else '⚠️  Not on Linux'} (GPU virtualization, KVM/Xen)")
        print(f"   NMDA Analyzer: {'✅ Available' if self.nmda.is_available() else '⚠️  Dependencies missing'} (NMDA antidepressant research)")
        print(f"   NPS Analyzer: {'✅ Available' if self.nps.is_available() else '⚠️  Dependencies missing'} (abuse potential, threat intel)")
        print(f"   OpenAI Pro: {'✅ Available' if self.openai.is_available() else '⚠️  Not configured'} (explicit request only)")

    def query(self, prompt, force_backend=None, images=None, video=None, **kwargs):
        """
        Unified query interface with LOCAL-FIRST routing

        Args:
            prompt: User query
            force_backend: "local", "gemini", "openai" (explicit override)
            images: List of image paths (triggers Gemini)
            video: Video path (triggers Gemini)
            **kwargs: Additional args (model preference, etc.)

        Returns:
            Unified response with backend info and routing decision
        """
        start_time = time.time()

        # Use Smart Router to determine best model
        routing_decision = self.router.route(
            prompt,
            has_images=bool(images or video),
            user_preference=force_backend
        )

        # Extract routing info
        selected_model = routing_decision['model']
        routing_reason = routing_decision['reason']
        routing_explanation = routing_decision['explanation']
        needs_web_search = routing_decision.get('web_search', False)

        # Determine backend (cloud vs local vs specialized)
        if selected_model == "gemini-pro":
            backend = "gemini"
        elif selected_model == "notebooklm":
            backend = "notebooklm"
        elif selected_model == "geospatial":
            backend = "geospatial"
        elif selected_model == "rdkit":
            backend = "rdkit"
        elif selected_model == "prt":
            backend = "prt"
        elif selected_model == "mxgpu":
            backend = "mxgpu"
        elif selected_model == "nmda":
            backend = "nmda"
        elif selected_model == "nps":
            backend = "nps"
        elif selected_model == "pharmaceutical":
            backend = "pharmaceutical"
        elif selected_model in ["gpt-4-turbo", "gpt-3.5-turbo"]:
            backend = "openai"
        else:
            backend = "local"  # All coding and general models are local

        # Execute on selected backend
        result = {}

        if backend == "local":
            # Check if Shodan search is needed (cybersecurity/threat intel queries)
            needs_shodan = self._needs_shodan_search(prompt)

            if needs_shodan:
                # Perform Shodan search
                shodan_results = self._perform_shodan_search(prompt)

                if 'error' not in shodan_results:
                    # Enhance prompt with Shodan results
                    shodan_context = self._format_shodan_results(shodan_results)
                    enhanced_prompt = f"{shodan_context}\n\nOriginal query: {prompt}\n\nProvide threat intelligence analysis based on Shodan results above."

                    # Generate response with Shodan context
                    result = self.local.generate(enhanced_prompt, model_selection=selected_model)
                    result['shodan_search'] = {
                        'performed': True,
                        'query': shodan_results['query'],
                        'facet': shodan_results['facet'],
                        'result_count': shodan_results['count']
                    }
                    result['web_search'] = {'performed': False}
                else:
                    # Shodan search failed, continue without it
                    result = self.local.generate(prompt, model_selection=selected_model)
                    result['shodan_search'] = {'performed': False, 'error': shodan_results.get('error')}
                    result['web_search'] = {'performed': False}

            # Check if web search is needed
            elif needs_web_search and self.web.ddgs_available:
                # Perform web search first
                search_results = self.web.search(prompt, max_results=5)

                if 'error' not in search_results:
                    # Enhance prompt with web results
                    web_context = self._format_web_results(search_results)
                    enhanced_prompt = f"{web_context}\n\nOriginal query: {prompt}\n\nProvide answer based on search results above."

                    # Generate response with web context
                    result = self.local.generate(enhanced_prompt, model_selection=selected_model)
                    result['web_search'] = {
                        'performed': True,
                        'source': search_results['source'],
                        'result_count': search_results['count'],
                        'urls': [r['url'] for r in search_results['results']]
                    }
                    result['shodan_search'] = {'performed': False}
                else:
                    # Web search failed, continue without it
                    result = self.local.generate(prompt, model_selection=selected_model)
                    result['web_search'] = {'performed': False, 'error': search_results.get('error')}
                    result['shodan_search'] = {'performed': False}
            else:
                # No web search needed or unavailable
                result = self.local.generate(prompt, model_selection=selected_model)
                result['web_search'] = {'performed': False}
                result['shodan_search'] = {'performed': False}

            # Add metadata
            result['backend'] = 'local'
            result['cost'] = 0.0
            result['privacy'] = 'local'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': selected_model,
                'reason': routing_reason,
                'explanation': routing_explanation,
                'emoji_tag': self.router.explain_routing(routing_decision, format='emoji')
            }

        elif backend == "gemini":
            # Try Gemini for multimodal
            local_response = None
            if not images and not video:
                # Get local response as fallback for non-multimodal
                local_result = self.local.generate(prompt)
                local_response = local_result.get('response')

            result = self.gemini.query(prompt, images=images, video=video, fallback_response=local_response)
            result['dsmil_attested'] = False

        elif backend == "notebooklm":
            # NotebookLM for document research
            task_mode = routing_decision.get('task_mode', 'qa')

            # Extract source management commands from kwargs
            if 'add_source' in kwargs:
                # Adding a source
                result = self.notebooklm.add_source(**kwargs['add_source'])
            elif 'create_notebook' in kwargs:
                # Creating a notebook
                result = self.notebooklm.create_notebook(**kwargs['create_notebook'])
            elif 'list_sources' in kwargs and kwargs['list_sources']:
                # List sources
                result = self.notebooklm.list_sources()
            elif 'list_notebooks' in kwargs and kwargs['list_notebooks']:
                # List notebooks
                result = self.notebooklm.list_notebooks()
            else:
                # Query mode
                result = self.notebooklm.query(
                    prompt=prompt,
                    mode=task_mode,
                    source_ids=kwargs.get('source_ids'),
                    notebook_id=kwargs.get('notebook_id')
                )

            result['backend'] = 'notebooklm'
            result['dsmil_attested'] = False
            result['routing'] = {
                'selected_model': 'notebooklm',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "geospatial":
            # Geospatial analytics
            task_mode = routing_decision.get('task_mode', 'create_map')

            # Route to appropriate method based on task mode
            if task_mode == 'load_data' and 'file_path' in kwargs:
                result = self.geospatial.load_data(**kwargs)
            elif task_mode in ['create_map', 'threat_intel', 'infrastructure']:
                if 'dataset_ids' in kwargs:
                    result = self.geospatial.create_map(**kwargs)
                else:
                    result = self.geospatial.list_datasets()
            else:
                # General query - provide status
                result = self.geospatial.get_status()
                result['message'] = "Geospatial agent ready. Use load_data() to import KML/GeoJSON/Shapefile"

            result['backend'] = 'geospatial'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'geospatial',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "rdkit":
            # RDKit cheminformatics
            task_mode = routing_decision.get('task_mode', 'parse')

            # Route to appropriate method
            if task_mode == 'parse' and 'structure' in kwargs:
                result = self.rdkit.parse_molecule(**kwargs)
            elif task_mode == 'descriptors' and 'mol_id' in kwargs:
                result = self.rdkit.calculate_descriptors(**kwargs)
            elif task_mode == 'fingerprint' and 'mol_id' in kwargs:
                result = self.rdkit.generate_fingerprint(**kwargs)
            elif task_mode == 'similarity' and 'query_mol_id' in kwargs:
                result = self.rdkit.similarity_search(**kwargs)
            elif task_mode == 'drug_likeness' and 'mol_id' in kwargs:
                result = self.rdkit.drug_likeness_analysis(**kwargs)
            elif task_mode == 'substructure' and 'pattern' in kwargs:
                result = self.rdkit.substructure_search(**kwargs)
            else:
                # General query - provide status
                result = self.rdkit.get_status()
                result['message'] = "RDKit agent ready. Use parse_molecule() to load molecular structures"

            result['backend'] = 'rdkit'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'rdkit',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "prt":
            # PRT data visualization
            task_mode = routing_decision.get('task_mode', 'visualize')

            # Route to appropriate method
            if task_mode == 'load' and 'file_path' in kwargs:
                result = self.prt.load_dataset(**kwargs)
            elif task_mode == 'visualize' and 'dataset_id' in kwargs:
                result = self.prt.visualize_dataset(**kwargs)
            elif task_mode == 'explore' and 'dataset_id' in kwargs:
                result = self.prt.explore_dataset(**kwargs)
            elif task_mode == 'classify' and 'dataset_id' in kwargs:
                result = self.prt.train_classifier(**kwargs)
            elif task_mode == 'cluster' and 'dataset_id' in kwargs:
                result = self.prt.cluster_analysis(**kwargs)
            elif task_mode == 'reduce' and 'dataset_id' in kwargs:
                result = self.prt.dimensionality_reduction(**kwargs)
            else:
                # General query - provide status
                result = self.prt.get_status()
                result['message'] = "PRT visualization agent ready. Use load_dataset() to import data"

            result['backend'] = 'prt'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'prt',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "mxgpu":
            # MxGPU GPU virtualization
            task_mode = routing_decision.get('task_mode', 'detect')

            # Route to appropriate method
            if task_mode == 'detect':
                result = self.mxgpu.detect_gpus()
            elif task_mode == 'sriov' and 'pci_id' in kwargs:
                result = self.mxgpu.get_sriov_status(**kwargs)
            elif task_mode == 'passthrough':
                result = self.mxgpu.get_iommu_groups()
            elif task_mode == 'config' and 'vm_name' in kwargs and 'gpu_pci_id' in kwargs:
                result = self.mxgpu.generate_vm_config(**kwargs)
            elif task_mode == 'status':
                result = self.mxgpu.check_vfio_status()
            else:
                # General query - provide status
                result = self.mxgpu.get_status()

            result['backend'] = 'mxgpu'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'mxgpu',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "nmda":
            # NMDA agonist antidepressant analysis
            task_mode = routing_decision.get('task_mode', 'nmda_activity')

            # Route to appropriate method
            if task_mode == 'nmda_activity' and 'mol_id' in kwargs:
                result = self.nmda.analyze_nmda_activity(**kwargs)
            elif task_mode == 'bbb_prediction' and 'mol_id' in kwargs:
                result = self.nmda.predict_bbb_penetration(**kwargs)
            elif task_mode == 'compare' and 'mol_id' in kwargs:
                result = self.nmda.compare_with_known_antidepressants(**kwargs)
            elif task_mode == 'comprehensive' and 'mol_id' in kwargs:
                result = self.nmda.comprehensive_analysis(**kwargs)
            else:
                # General query - provide status
                result = self.nmda.get_status()
                result['message'] = "NMDA analyzer ready. Parse molecules with RDKit first, then analyze"

            result['backend'] = 'nmda'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'nmda',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "nps":
            # NPS abuse potential analysis
            task_mode = routing_decision.get('task_mode', 'abuse_potential')

            # Route to appropriate method
            if task_mode == 'classify' and 'mol_id' in kwargs:
                result = self.nps.classify_nps(**kwargs)
            elif task_mode == 'abuse_potential' and 'mol_id' in kwargs:
                comprehensive = kwargs.get('comprehensive', False)
                result = self.nps.predict_abuse_potential(mol_id=kwargs['mol_id'], comprehensive=comprehensive)
            elif task_mode == 'receptor_binding' and 'mol_id' in kwargs:
                result = self.nps.predict_receptor_binding(**kwargs)
            elif task_mode == 'batch' and 'mol_ids' in kwargs:
                result = self.nps.batch_screening(**kwargs)
            else:
                # General query - provide status
                result = self.nps.get_status()
                result['message'] = "NPS analyzer ready. Parse molecules with RDKit first, then analyze"

            result['backend'] = 'nps'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'nps',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode
            }

        elif backend == "pharmaceutical":
            # Pharmaceutical research corpus (NMDA + NPS + ZEROPAIN)
            task_mode = routing_decision.get('task_mode', 'screen')

            # Check if this is a conversational query (natural language without structured kwargs)
            use_conversational = 'smiles' not in kwargs and 'mol_id' not in kwargs and 'compound_protocol' not in kwargs

            if use_conversational:
                # Use conversational agent for context-aware natural language processing
                try:
                    raw_result, conversational_response = self.pharma_conversation.process_conversational_query(prompt)
                    result = raw_result
                    result['conversational_response'] = conversational_response
                    result['conversation_turn'] = len(self.pharma_conversation.conversation_history)
                    result['suggestions'] = self.pharma_conversation.conversation_history[-1].suggestions if self.pharma_conversation.conversation_history else []
                    result['session_id'] = self.pharma_conversation.session_id
                except Exception as e:
                    result = {'status': 'error', 'message': f'Conversational processing failed: {e}'}
            else:
                # Structured API mode - route to appropriate method
                if task_mode == 'screen' and 'smiles' in kwargs:
                    result = self.pharmaceutical.screen_compound(**kwargs)
                elif task_mode == 'docking' and 'smiles' in kwargs:
                    result = self.pharmaceutical.dock_to_receptors(**kwargs)
                elif task_mode == 'admet' and 'smiles' in kwargs:
                    result = self.pharmaceutical.predict_admet(**kwargs)
                elif task_mode == 'safety' and 'smiles' in kwargs:
                    result = self.pharmaceutical.comprehensive_safety_profile(**kwargs)
                elif task_mode == 'simulation' and 'compound_protocol' in kwargs:
                    result = self.pharmaceutical.simulate_patients(**kwargs)
                elif task_mode == 'zeropain':
                    result = {
                        'status': 'success',
                        'message': 'ZEROPAIN pain management protocols integrated',
                        'capabilities': ['opioid_optimization', 'patient_simulation', 'pk_pd_modeling'],
                        'info': 'Use patient simulation with multi-compound protocols for zero-tolerance therapy'
                    }
                elif task_mode == 'nmda_antidepressant' and 'mol_id' in kwargs:
                    # Delegate to NMDA analyzer for antidepressant analysis
                    result = self.nmda.comprehensive_analysis(**kwargs)
                elif task_mode == 'nps_abuse' and 'mol_id' in kwargs:
                    # Delegate to NPS analyzer for abuse potential
                    result = self.nps.predict_abuse_potential(**kwargs)
                else:
                    # General query - provide status
                    result = {
                        'status': 'success',
                        'message': 'Pharmaceutical Research Corpus ready',
                        'capabilities': {
                            'discovery': ['compound_screening', 'therapeutic_classification'],
                            'validation': ['molecular_docking', 'admet_prediction', 'bbb_penetration'],
                            'safety': ['toxicity_assessment', 'abuse_potential', 'nps_classification'],
                            'optimization': ['patient_simulation', 'protocol_optimization', 'zeropain_therapy'],
                            'reporting': ['regulatory_dossier', 'safety_profile']
                        },
                        'tempest_level': self.pharmaceutical.tempest_level,
                        'integrations': ['NMDA', 'NPS', 'ZEROPAIN', 'Intel_AI', 'AutoDock_Vina']
                    }

            result['backend'] = 'pharmaceutical'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': 'pharmaceutical',
                'reason': routing_reason,
                'explanation': routing_explanation,
                'task_mode': task_mode,
                'conversational_mode': use_conversational
            }

        elif backend == "openai":
            # Try OpenAI (explicit request only)
            local_result = self.local.generate(prompt)
            local_response = local_result.get('response')

            result = self.openai.query(prompt, model=kwargs.get('model', 'gpt-4-turbo'), fallback_response=local_response)
            result['dsmil_attested'] = False

        else:
            # Unknown backend → local
            result = self.local.generate(prompt)
            result['backend'] = 'local_deepseek'
            result['cost'] = 0.0
            result['privacy'] = 'local'
            result['dsmil_attested'] = True
            routing_reason = "unknown_backend_fallback"

        # Add enhanced routing metadata
        if 'routing' not in result:
            result['routing'] = {
                'selected_model': selected_model,
                'reason': routing_reason,
                'explanation': routing_explanation
            }

        result['routed_to'] = backend
        result['total_time'] = round(time.time() - start_time, 2)
        result['timestamp'] = time.time()

        # Convert to Pydantic model if in Pydantic mode
        if self.pydantic_mode and PYDANTIC_AVAILABLE:
            return self._dict_to_pydantic_response(result, backend, selected_model, start_time)

        return result

    def _format_web_results(self, search_results):
        """Format web search results for AI context"""
        context = f"Web search results for: {search_results['query']}\n\n"

        for i, result in enumerate(search_results['results'], 1):
            context += f"[{i}] {result['title']}\n"
            context += f"    {result['snippet']}\n"
            context += f"    URL: {result['url']}\n\n"

        return context

    def _needs_shodan_search(self, prompt: str) -> bool:
        """Detect if prompt requires Shodan search (cybersecurity/threat intel)"""
        prompt_lower = prompt.lower()

        # Keywords that indicate Shodan search is appropriate
        shodan_keywords = [
            'cve-', 'vulnerability', 'vulnerabilities', 'exploit', 'shodan',
            'exposed', 'honeypot', 'compromised', 'iot devices', 'open ports',
            'attack surface', 'threat intelligence', 'security scan',
            'network scan', 'internet scan', 'ip address', 'internet-facing',
            'publicly accessible', 'shodan search', 'find vulnerable',
            'devices running', 'servers running', 'apache', 'nginx', 'iis',
            'ssh', 'rdp', 'ftp', 'telnet', 'scada', 'industrial control'
        ]

        return any(keyword in prompt_lower for keyword in shodan_keywords)

    def _perform_shodan_search(self, prompt: str) -> dict:
        """Perform Shodan search based on prompt analysis"""
        prompt_lower = prompt.lower()

        # Extract CVE if mentioned
        import re
        cve_match = re.search(r'cve-\d{4}-\d{4,7}', prompt_lower)

        if cve_match:
            cve_id = cve_match.group(0).upper()
            return self.shodan.search_vulnerability(cve_id, facet='country')

        # Check for honeypot queries
        if 'honeypot' in prompt_lower:
            return self.shodan.search_honeypots()

        # Check for compromised systems
        if 'compromised' in prompt_lower:
            return self.shodan.search_compromised()

        # Check for product/service queries
        products = ['apache', 'nginx', 'iis', 'tomcat', 'jenkins', 'redis', 'mongodb', 'elasticsearch']
        for product in products:
            if product in prompt_lower:
                return self.shodan.search_product(product, facet='version')

        # Check for port queries
        port_match = re.search(r'port[:\s]+(\d+)', prompt_lower)
        if port_match:
            port = int(port_match.group(1))
            return self.shodan.search_port(port)

        # Default: general search with extracted terms
        # Remove common words and use remaining as query
        query_terms = re.sub(r'\b(search|find|show|what|how|many|are|is|the|for|with)\b', '', prompt_lower)
        query_terms = ' '.join(query_terms.split())

        if query_terms:
            return self.shodan.search(query_terms, facet='country')
        else:
            return {'error': 'Could not extract search terms from query'}

    def _format_shodan_results(self, search_results: dict) -> str:
        """Format Shodan search results for AI context"""
        if 'error' in search_results:
            return f"Shodan search error: {search_results['error']}"

        context = f"Shodan Search Results\n"
        context += f"Query: {search_results['query']}\n"
        context += f"Grouped by: {search_results['facet_name']}\n"
        context += f"Total results: {search_results['count']}\n\n"
        context += "Top findings:\n\n"

        for i, result in enumerate(search_results['results'][:20], 1):
            context += f"[{i}] {result['value']}: {result['count']} instances\n"

        if search_results['count'] > 20:
            context += f"\n... and {search_results['count'] - 20} more results\n"

        context += f"\nSource: {search_results['source']} ({search_results['privacy']})\n"

        return context

    def execute_workflow(self, task_description, task_type="feature",
                        complexity="medium", constraints=None,
                        model_preference="quality_code"):
        """
        Execute ACE-FCA phase-based workflow for complex coding tasks

        Workflow: Research → Plan → Implement → Verify
        Each phase includes:
        - Context isolation via specialized subagents
        - Automatic compaction at 40-60% utilization
        - Human review checkpoints at phase boundaries

        Args:
            task_description: Description of the coding task
            task_type: 'feature', 'bugfix', 'refactor', 'analysis'
            complexity: 'simple', 'medium', 'complex'
            constraints: List of constraints (backward compat, etc.)
            model_preference: Model to use ('fast', 'code', 'quality_code', etc.)

        Returns:
            Dict with workflow results and phase outputs
        """
        if not self.ace_enabled:
            return {
                "success": False,
                "error": "ACE-FCA not available. Install ace_context_engine module."
            }

        # Create workflow task
        task = WorkflowTask(
            description=task_description,
            task_type=task_type,
            estimated_complexity=complexity,
            constraints=constraints or []
        )

        # Execute workflow
        result = self.ace_workflow.execute_task(task, model_preference=model_preference)

        return result

    def use_subagent(self, agent_type, task_params):
        """
        Use specialized subagent for context-isolated task execution

        Available agents:
        - 'research': Codebase exploration and analysis
        - 'planner': Implementation planning
        - 'implementer': Code generation
        - 'verifier': Testing and validation
        - 'summarizer': Content compression

        Args:
            agent_type: Type of subagent to use
            task_params: Dict of parameters for the subagent

        Returns:
            SubagentResult with compressed output
        """
        if not self.ace_enabled:
            return {
                "success": False,
                "error": "ACE-FCA not available"
            }

        try:
            agent = create_subagent(agent_type, self.local)
            result = agent.execute(task_params)
            return {
                "success": result.success,
                "compressed_output": result.compressed_output,
                "metadata": result.metadata,
                "error": result.error
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_context_stats(self):
        """Get ACE-FCA context window statistics"""
        if not self.ace_enabled or not self.ace_engine:
            return {"error": "ACE-FCA not enabled"}

        return self.ace_engine.get_stats()

    def compact_context(self, target_tokens=None):
        """
        Manually trigger context compaction

        Args:
            target_tokens: Target token count (default: 50% of max)

        Returns:
            Dict with compaction statistics
        """
        if not self.ace_enabled or not self.ace_engine:
            return {"error": "ACE-FCA not enabled"}

        return self.ace_engine.compact_context(target_tokens)

    def get_status(self):
        """Get comprehensive status of all backends"""
        local_status = self.local.get_status()

        status = {
            "backends": {
                "local_deepseek": {
                    "available": True,  # Always available
                    "priority": "PRIMARY (default for all queries)",
                    "models": local_status.get('models', {}),
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "gemini_pro": {
                    "available": self.gemini.is_available(),
                    "priority": "MULTIMODAL ONLY (images/video)",
                    "model": "gemini-2.0-flash-exp" if self.gemini.is_available() else "not_configured",
                    "student_edition": True,
                    "dsmil_attested": False,
                    "cost_per_query": 0,  # Student free tier
                    "privacy": "cloud"
                },
                "notebooklm": {
                    "available": self.notebooklm.is_available(),
                    "priority": "DOCUMENT RESEARCH (auto-detected)",
                    "model": "gemini-2.0-flash-exp" if self.notebooklm.is_available() else "not_configured",
                    "dsmil_attested": False,
                    "cost_per_query": 0,  # Student free tier
                    "privacy": "cloud",
                    "capabilities": [
                        "Document ingestion and source management",
                        "Multi-source Q&A with grounding",
                        "Summary, FAQ, study guide generation",
                        "Source synthesis and analysis",
                        "Executive briefings"
                    ]
                },
                "openai_pro": {
                    "available": self.openai.is_available(),
                    "priority": "EXPLICIT REQUEST ONLY (not auto-routed)",
                    "models": ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"],
                    "dsmil_attested": False,
                    "cost_per_1k_tokens": "$0.02",
                    "privacy": "cloud"
                },
                "geospatial": {
                    "available": self.geospatial.is_available(),
                    "priority": "AUTO-ROUTED (OSINT, threat-intel, infrastructure mapping)",
                    "capabilities": [
                        "Geospatial data loading (KML, GeoJSON, Shapefile, GPX)",
                        "Interactive map creation (2D/3D)",
                        "Threat intelligence mapping",
                        "Infrastructure visualization",
                        "Temporal analytics"
                    ],
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "rdkit": {
                    "available": self.rdkit.is_available(),
                    "priority": "AUTO-ROUTED (cheminformatics, drug discovery)",
                    "capabilities": [
                        "Molecular structure parsing (SMILES, SDF, MOL)",
                        "Descriptor calculation (>200 descriptors)",
                        "Fingerprint generation (Morgan, MACCS, RDK)",
                        "Similarity/substructure searching",
                        "Drug-likeness analysis (Lipinski, Veber)"
                    ],
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "prt_visualization": {
                    "available": self.prt.is_available(),
                    "priority": "AUTO-ROUTED (data viz, ML, pattern recognition)",
                    "capabilities": [
                        "Interactive dataset exploration",
                        "Classification/clustering workflows",
                        "Dimensionality reduction (PCA, t-SNE)",
                        "ML model visualization",
                        "Pattern recognition pipelines"
                    ],
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "mxgpu": {
                    "available": self.mxgpu.is_available(),
                    "priority": "AUTO-ROUTED (GPU virtualization, KVM/Xen)",
                    "capabilities": [
                        "GPU SR-IOV detection and configuration",
                        "KVM/Xen GPU passthrough setup",
                        "IOMMU group management",
                        "VM configuration generation",
                        "VF allocation and monitoring"
                    ],
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "nmda_analyzer": {
                    "available": self.nmda.is_available(),
                    "priority": "AUTO-ROUTED (NMDA antidepressant research)",
                    "capabilities": [
                        "NMDA receptor activity prediction",
                        "Blood-Brain Barrier (BBB) penetration analysis",
                        "Comparison with ketamine/esketamine/memantine",
                        "Antidepressant potential scoring",
                        "Novel compound assessment",
                        "Safety warning generation"
                    ],
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "nps_analyzer": {
                    "available": self.nps.is_available(),
                    "priority": "AUTO-ROUTED (NPS abuse potential, threat intelligence)",
                    "capabilities": [
                        "Novel psychoactive substance classification",
                        "Abuse potential prediction (0-10 scale)",
                        "Receptor binding prediction (6 neurotransmitter systems)",
                        "Neurotoxicity and lethality assessment",
                        "Antidote recommendations",
                        "Dark web proliferation prediction",
                        "DEA scheduling recommendations",
                        "Comprehensive 12-hour analysis mode",
                        "Batch screening (1M+ compounds)"
                    ],
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                }
            },
            "routing_philosophy": "LOCAL-FIRST + SPECIALIZED AGENTS",
            "default_backend": "local_deepseek",
            "specialized_agents": {
                "geospatial": self.geospatial.is_available(),
                "rdkit": self.rdkit.is_available(),
                "prt": self.prt.is_available(),
                "mxgpu": self.mxgpu.is_available(),
                "nmda": self.nmda.is_available(),
                "nps": self.nps.is_available()
            },
            "dsmil": local_status.get('dsmil', {}),
            "total_compute": "76.4 TOPS (NPU 26.4 + GPU 40 + NCS2 10)"
        }

        # Add ACE-FCA status
        if self.ace_enabled:
            status["ace_fca"] = {
                "available": True,
                "features": [
                    "Context compaction (40-60% utilization)",
                    "Phase-based workflows (Research→Plan→Implement→Verify)",
                    "Specialized subagents with context isolation",
                    "Human review checkpoints at compaction boundaries"
                ],
                "context_stats": self.get_context_stats() if self.ace_engine else {}
            }
        else:
            status["ace_fca"] = {
                "available": False,
                "reason": "Module not loaded"
            }

        # Add parallel execution status
        if self.parallel_enabled:
            parallel_status = self.parallel_executor.get_status() if self.parallel_executor else {}
            worktree_stats = self.worktree_manager.get_stats() if self.worktree_manager else {}
            distributor_stats = self.task_distributor.get_system_status() if self.task_distributor else {}

            status["parallel_execution"] = {
                "available": True,
                "features": [
                    "M U L T I C L A U D E - Run multiple AI agents simultaneously",
                    "Git worktree management for parallel development",
                    "Intelligent task distribution across agents",
                    "Concurrent workflow execution"
                ],
                "executor": parallel_status,
                "worktrees": worktree_stats,
                "task_distribution": distributor_stats
            }
        else:
            status["parallel_execution"] = {
                "available": False,
                "reason": "Module not loaded"
            }

        return status

    def _dict_to_pydantic_response(self, result_dict, backend, selected_model, start_time):
        """Convert dict result to Pydantic OrchestratorResponse"""
        if not PYDANTIC_AVAILABLE:
            return result_dict

        try:
            # Extract routing info
            routing_info = result_dict.get('routing', {})

            # Map backend string to BackendType enum
            backend_map = {
                'local': BackendType.LOCAL,
                'gemini': BackendType.GEMINI,
                'openai': BackendType.OPENAI,
                'notebooklm': BackendType.NOTEBOOKLM,
                'geospatial': BackendType.GEOSPATIAL,
                'rdkit': BackendType.RDKIT,
                'prt': BackendType.PRT,
                'mxgpu': BackendType.MXGPU,
                'nmda': BackendType.NMDA,
                'nps': BackendType.NPS,
                'pharmaceutical': BackendType.PHARMACEUTICAL,
            }
            backend_enum = backend_map.get(backend, BackendType.LOCAL)

            # Map routing reason string to enum
            reason_str = routing_info.get('reason', 'general_query')
            reason_map = {
                'code_query': RoutingReason.CODE_QUERY,
                'general_query': RoutingReason.GENERAL_QUERY,
                'complex_reasoning': RoutingReason.COMPLEX_REASONING,
                'multimodal': RoutingReason.MULTIMODAL,
                'specialized_domain': RoutingReason.SPECIALIZED_DOMAIN,
                'user_preference': RoutingReason.USER_PREFERENCE,
                'web_search_needed': RoutingReason.WEB_SEARCH_NEEDED,
                'threat_intelligence': RoutingReason.THREAT_INTELLIGENCE,
            }
            reason_enum = reason_map.get(reason_str, RoutingReason.GENERAL_QUERY)

            # Create RoutingDecision
            routing_decision = RoutingDecision(
                selected_model=routing_info.get('selected_model', selected_model),
                backend=backend_enum,
                reason=reason_enum,
                explanation=routing_info.get('explanation', 'Auto-routed query'),
                confidence=routing_info.get('confidence', 0.8),
                web_search_needed=result_dict.get('web_search', {}).get('performed', False),
                shodan_search_needed=result_dict.get('shodan_search', {}).get('performed', False)
            )

            # Create WebSearchMeta
            web_search_dict = result_dict.get('web_search', {})
            web_search_results = []
            if 'results' in web_search_dict:
                for i, r in enumerate(web_search_dict['results']):
                    web_search_results.append(WebSearchResult(
                        title=r.get('title', ''),
                        url=r.get('url', ''),
                        snippet=r.get('snippet', ''),
                        position=i + 1
                    ))

            web_search = WebSearchMeta(
                performed=web_search_dict.get('performed', False),
                source=web_search_dict.get('source'),
                result_count=web_search_dict.get('result_count', 0),
                urls=web_search_dict.get('urls', []),
                results=web_search_results,
                error=web_search_dict.get('error')
            )

            # Create ShodanSearchMeta
            shodan_search_dict = result_dict.get('shodan_search', {})
            shodan_search = ShodanSearchMeta(
                performed=shodan_search_dict.get('performed', False),
                query=shodan_search_dict.get('query'),
                facet=shodan_search_dict.get('facet'),
                result_count=shodan_search_dict.get('result_count', 0),
                error=shodan_search_dict.get('error')
            )

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Get response text
            response_text = result_dict.get('response', '')
            if isinstance(response_text, dict):
                # Some backends return nested response
                response_text = response_text.get('response', str(response_text))

            # Create OrchestratorResponse
            return OrchestratorResponse(
                response=response_text,
                backend=backend_enum,
                selected_model=selected_model,
                routing=routing_decision,
                web_search=web_search,
                shodan_search=shodan_search,
                latency_ms=latency_ms,
                cost=result_dict.get('cost', 0.0),
                privacy=result_dict.get('privacy', 'local'),
                dsmil_attested=result_dict.get('dsmil_attested', False),
                attestation_hash=result_dict.get('attestation_hash'),
                error=result_dict.get('error'),
                fallback_used=result_dict.get('fallback_used', False)
            )

        except Exception as e:
            # If Pydantic conversion fails, return dict
            print(f"Warning: Pydantic conversion failed: {e}")
            return result_dict


# CLI
if __name__ == "__main__":
    import json

    orchestrator = UnifiedAIOrchestrator()

    if len(sys.argv) < 2:
        print("\nDSMIL Unified Orchestrator - Usage:")
        print("  python3 unified_orchestrator.py status")
        print("  python3 unified_orchestrator.py query 'your question'")
        print("  python3 unified_orchestrator.py query 'your question' --backend gemini")
        print("  python3 unified_orchestrator.py query 'your question' --backend openai")
        print("  python3 unified_orchestrator.py image 'describe this' /path/to/image.jpg")
        print("  python3 unified_orchestrator.py workflow 'task description' [--type feature] [--complexity medium]")
        print("\nLOCAL-FIRST: Defaults to local DeepSeek for privacy and no guardrails")
        print("ACE-FCA: Phase-based workflows with context compaction and human review")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))

    elif cmd == "query" and len(sys.argv) > 2:
        query = sys.argv[2]
        backend = None

        # Check for --backend flag
        if '--backend' in sys.argv:
            idx = sys.argv.index('--backend')
            if idx + 1 < len(sys.argv):
                backend = sys.argv[idx + 1]

        print(f"\n🎯 Query: {query}")
        result = orchestrator.query(query, force_backend=backend)

        print(f"\n{'='*60}")
        print(result['response'])
        print(f"{'='*60}")
        print(f"Backend: {result['routed_to']}")
        if 'routing' in result:
            print(f"Routing: {result['routing']['emoji_tag']}")
        print(f"Model: {result.get('model', 'N/A')}")
        if 'inference_time' in result:
            print(f"Time: {result['inference_time']}s")
        print(f"Cost: ${result.get('cost', 0):.4f}")
        print(f"Privacy: {result.get('privacy', 'N/A')}")
        print(f"DSMIL Attested: {result.get('dsmil_attested', False)}")

    elif cmd == "image" and len(sys.argv) > 3:
        query = sys.argv[2]
        images = sys.argv[3:]

        print(f"\n🎯 Multimodal Query: {query}")
        print(f"   Images: {len(images)}")

        result = orchestrator.query(query, images=images)
        print(json.dumps(result, indent=2))

    elif cmd == "workflow" and len(sys.argv) > 2:
        task_desc = sys.argv[2]
        task_type = "feature"
        complexity = "medium"

        # Parse optional flags
        if '--type' in sys.argv:
            idx = sys.argv.index('--type')
            if idx + 1 < len(sys.argv):
                task_type = sys.argv[idx + 1]

        if '--complexity' in sys.argv:
            idx = sys.argv.index('--complexity')
            if idx + 1 < len(sys.argv):
                complexity = sys.argv[idx + 1]

        print(f"\n🔄 ACE-FCA Workflow: {task_desc}")
        print(f"   Type: {task_type}")
        print(f"   Complexity: {complexity}")
        print(f"   Phases: Research → Plan → Implement → Verify")

        result = orchestrator.execute_workflow(
            task_description=task_desc,
            task_type=task_type,
            complexity=complexity
        )

        if result.get('success'):
            print(f"\n{'='*60}")
            print("WORKFLOW COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"\nPhases completed: {', '.join(result['phases_completed'])}")
            print(f"\n📊 Context Stats:")
            for key, val in result['context_stats'].items():
                print(f"   {key}: {val}")

            if 'review_checkpoints' in result:
                print(f"\n✓ Review checkpoints: {len(result['review_checkpoints'])}")

        else:
            print(f"\n✗ Workflow failed: {result.get('error', 'Unknown error')}")
            if 'phase_stopped' in result:
                print(f"   Stopped at phase: {result['phase_stopped']}")
