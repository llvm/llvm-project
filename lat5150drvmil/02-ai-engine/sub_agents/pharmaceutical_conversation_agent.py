#!/usr/bin/env python3
"""
Pharmaceutical Conversation Agent
Context-aware conversational AI for pharmaceutical research

Integrates with existing NLI to provide:
- Multi-turn pharmaceutical research conversations
- Context management across queries
- Smart suggestions based on previous results
- Automatic compound library management
- TEMPEST-aware workflow guidance

Author: LAT5150DRVMIL Integration
Version: 1.0.0
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from .pharmaceutical_corpus import PharmaceuticalCorpus, TEMPESTLevel
from .rdkit_wrapper import RDKitAgent
from .nmda_agonist_analyzer import NMDAAgonistAnalyzer
from .nps_abuse_potential_analyzer import NPSAbusePotentialAnalyzer


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: str
    user_query: str
    intent: str
    compound: Optional[str]
    action: str
    result: Dict[str, Any]
    suggestions: List[str] = field(default_factory=list)
    tempest_level: int = 1


@dataclass
class CompoundContext:
    """Context for a compound being analyzed"""
    smiles: str
    name: Optional[str]
    mol_id: Optional[str]
    analyses_performed: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_analyzed: str = field(default_factory=lambda: datetime.now().isoformat())


class PharmaceuticalConversationAgent:
    """
    Conversational AI agent for pharmaceutical research

    Manages multi-turn conversations with context awareness,
    automatic workflow suggestions, and intelligent routing.
    """

    def __init__(self, tempest_level: int = 1, user_id: str = "conversation_user"):
        """
        Initialize conversation agent

        Args:
            tempest_level: TEMPEST security level (0-3)
            user_id: User identifier for session tracking
        """
        self.tempest_level = tempest_level
        self.user_id = user_id

        # Initialize pharmaceutical agents
        self.corpus = PharmaceuticalCorpus(tempest_level=tempest_level, user_id=user_id)
        self.rdkit = RDKitAgent()
        self.nmda = NMDAAgonistAnalyzer()
        self.nps = NPSAbusePotentialAnalyzer(verbose=False)

        # Conversation state
        self.conversation_history: List[ConversationTurn] = []
        self.compound_contexts: Dict[str, CompoundContext] = {}  # smiles -> context
        self.current_compound: Optional[str] = None
        self.session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Intent detection patterns
        self._init_intent_patterns()

    def _init_intent_patterns(self):
        """Initialize intent detection patterns"""
        self.intent_patterns = {
            'screen': r'(screen|analyze|test|evaluate|assess|check)',
            'dock': r'(dock|docking|bind|binding|affinity)',
            'admet': r'(admet|pharmacokinetic|pk/pd|absorption|distribution|metabolism|excretion)',
            'bbb': r'(bbb|blood.?brain|brain penetration)',
            'safety': r'(safety|safe|toxicity|toxic|adverse|side effect)',
            'abuse': r'(abuse|recreational|addiction|nps|novel psychoactive)',
            'nmda': r'(nmda|antidepressant|depression|glutamate|ketamine analog)',
            'simulation': r'(simulate|simulation|patient|clinical trial|protocol)',
            'compare': r'(compare|versus|vs|difference|similar)',
            'suggest': r'(suggest|recommend|what next|what should|guidance)',
            'explain': r'(explain|what is|tell me about|describe)',
            'status': r'(status|progress|summary|what have we)',
        }

    def detect_intent(self, query: str) -> str:
        """
        Detect user intent from natural language query

        Returns: Intent name (screen, dock, admet, etc.)
        """
        query_lower = query.lower()

        # Check for each intent pattern
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                return intent

        # Default to screen if no specific intent detected
        return 'screen'

    def extract_compound(self, query: str) -> Optional[str]:
        """
        Extract compound SMILES or name from query

        Returns: SMILES string or None
        """
        # Check for SMILES pattern
        smiles_pattern = r'[A-Z][A-Za-z0-9@\+\-\[\]\(\)\=\#]+(?:[A-Za-z0-9@\+\-\[\]\(\)\=\#]*)'
        smiles_match = re.search(smiles_pattern, query)

        if smiles_match:
            return smiles_match.group(0)

        # Check for common compound names
        compound_library = {
            'fentanyl': 'CCN(CC)C(=O)C1CN(C)CCc2ccccc21',
            'ketamine': 'CC(=O)C(c1ccccc1Cl)N(C)C',
            'esketamine': 'CC(=O)C(c1ccccc1Cl)N(C)C',
            'memantine': 'CC1(C)C2CCC1(C)NC2',
            'mdma': 'CC(CC1=CC2=C(C=C1)OCO2)NC',
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'cocaine': 'COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C',
            'morphine': 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O',
            'heroin': 'CC(=O)OC1C=CC2C3CC4=C5C2(C1OC(=O)C)C(=O)CCN4CC5=C(C=C3)O',
            'thc': 'CCCCCc1cc(O)c2c(c1)OC(C)(C)C1CCC(C)=CC21',
            'lsd': 'CCN(CC)C(=O)C1CN(C)C2Cc3c[nH]c4cccc(C2=C1)c34',
            'psilocybin': 'CN(C)CCC1=CNC2=C1C=C(C=C2)OP(=O)(O)O',
            'amphetamine': 'CC(N)Cc1ccccc1',
            'methamphetamine': 'CN(C)CC(C)c1ccccc1',
        }

        query_lower = query.lower()
        for name, smiles in compound_library.items():
            if name in query_lower:
                return smiles

        # Check if referencing current compound
        if any(word in query_lower for word in ['it', 'this', 'same', 'current', 'that']):
            if self.current_compound:
                return self.current_compound

        return None

    def get_context_suggestions(self, compound: str, completed_analyses: List[str]) -> List[str]:
        """
        Generate smart suggestions based on context

        Returns: List of suggested next steps
        """
        suggestions = []

        # Get compound context if exists
        context = self.compound_contexts.get(compound)
        if not context:
            return ["Start with: screen compound", "Or: classify therapeutic potential"]

        analyses = set(context.analyses_performed)

        # Suggest based on what's been done and TEMPEST level
        if 'screen' not in analyses:
            suggestions.append("Run comprehensive screening first")

        if 'screen' in analyses and 'admet' not in analyses and self.tempest_level >= 1:
            suggestions.append("Predict ADMET properties with Intel AI")

        if 'admet' in analyses and 'bbb' not in analyses and self.tempest_level >= 1:
            suggestions.append("Analyze Blood-Brain Barrier penetration")

        if 'screen' in analyses and 'dock' not in analyses and self.tempest_level >= 2:
            suggestions.append("Perform molecular docking to target receptors")

        if 'screen' in analyses and 'abuse' not in analyses and self.tempest_level >= 2:
            suggestions.append("Assess abuse potential (NPS classification)")

        if ('dock' in analyses or 'admet' in analyses) and 'safety' not in analyses and self.tempest_level >= 2:
            suggestions.append("Generate comprehensive safety profile")

        if 'safety' in analyses and 'simulation' not in analyses and self.tempest_level >= 3:
            suggestions.append("Run patient simulation (100k virtual patients)")

        if len(analyses) >= 3 and 'compare' not in analyses:
            suggestions.append("Compare with known pharmaceutical compounds")

        # If all basic analyses done, suggest advanced
        if len(analyses) >= 4:
            if self.tempest_level >= 3:
                suggestions.append("Generate regulatory submission dossier")
            else:
                suggestions.append(f"Upgrade to TEMPEST Level 3 for patient simulation and regulatory tools")

        return suggestions[:3]  # Return top 3 suggestions

    def build_conversational_response(
        self,
        result: Dict[str, Any],
        intent: str,
        compound: Optional[str]
    ) -> str:
        """
        Build conversational response with context and suggestions

        Returns: Natural language response string
        """
        response_parts = []

        # Handle errors
        if result.get('status') == 'error':
            response_parts.append(f"❌ {result.get('message', 'Unknown error')}")
            if 'suggestion' in result:
                response_parts.append(f"\n💡 {result['suggestion']}")
            return '\n'.join(response_parts)

        # Success - build contextual response
        if intent == 'screen':
            name = result.get('name', 'Compound')
            response_parts.append(f"✅ Screened {name}")

            if 'nps_classification' in result:
                nps = result['nps_classification']
                if nps.get('is_nps'):
                    response_parts.append(f"⚠️  Novel Psychoactive Substance detected: {nps.get('class', 'Unknown')}")

            if 'drug_likeness' in result:
                dl = result['drug_likeness']
                if dl.get('lipinski_pass'):
                    response_parts.append(f"✅ Passes Lipinski Rule of Five (drug-like)")
                else:
                    response_parts.append(f"⚠️  Violates Lipinski Rule of Five")

        elif intent == 'dock':
            response_parts.append(f"✅ Molecular docking complete")
            if 'docking_results' in result:
                best_affinity = min(
                    (data.get('binding_affinity', 999) for data in result['docking_results'].values()),
                    default=None
                )
                if best_affinity and best_affinity < -8.0:
                    response_parts.append(f"🎯 Strong binding detected ({best_affinity:.1f} kcal/mol)")

        elif intent == 'admet':
            response_parts.append(f"✅ ADMET analysis complete")
            if result.get('bbb_permeability') == 'HIGH':
                response_parts.append(f"🧠 High BBB penetration - CNS active")
            if result.get('bioavailability', 0) > 70:
                response_parts.append(f"💊 Good oral bioavailability ({result['bioavailability']:.0f}%)")

        elif intent == 'safety':
            score = result.get('safety_score', 0)
            if score >= 7:
                response_parts.append(f"✅ Good safety profile (score: {score:.1f}/10)")
            elif score >= 4:
                response_parts.append(f"⚠️  Moderate safety concerns (score: {score:.1f}/10)")
            else:
                response_parts.append(f"❌ Significant safety concerns (score: {score:.1f}/10)")

            if 'warnings' in result and result['warnings']:
                response_parts.append(f"⚠️  {len(result['warnings'])} warning(s) identified")

        elif intent == 'abuse':
            score = result.get('abuse_score', 0)
            if score >= 7:
                response_parts.append(f"🚨 HIGH abuse potential (score: {score:.1f}/10)")
            elif score >= 4:
                response_parts.append(f"⚠️  MODERATE abuse potential (score: {score:.1f}/10)")
            else:
                response_parts.append(f"✅ LOW abuse potential (score: {score:.1f}/10)")

        elif intent == 'nmda':
            response_parts.append(f"✅ NMDA antidepressant analysis complete")
            if 'nmda_activity' in result:
                activity = result['nmda_activity']
                if activity > 7:
                    response_parts.append(f"🎯 Strong NMDA activity ({activity:.1f}/10)")

        elif intent == 'status':
            n_compounds = len(self.compound_contexts)
            n_turns = len(self.conversation_history)
            response_parts.append(f"📊 Session Status:")
            response_parts.append(f"   • {n_compounds} compound(s) analyzed")
            response_parts.append(f"   • {n_turns} conversation turn(s)")
            response_parts.append(f"   • TEMPEST Level {self.tempest_level}")
            if self.current_compound:
                context = self.compound_contexts.get(self.current_compound)
                if context:
                    response_parts.append(f"   • Current: {context.name or 'Unknown'}")
                    response_parts.append(f"   • Analyses: {', '.join(context.analyses_performed)}")

        # Add suggestions
        if compound and intent != 'status':
            suggestions = self.get_context_suggestions(compound,
                self.compound_contexts.get(compound, CompoundContext(smiles=compound, name=None, mol_id=None)).analyses_performed
            )

            if suggestions:
                response_parts.append(f"\n💡 Suggested next steps:")
                for i, suggestion in enumerate(suggestions, 1):
                    response_parts.append(f"   {i}. {suggestion}")

        return '\n'.join(response_parts)

    def process_conversational_query(self, query: str) -> Tuple[Dict[str, Any], str]:
        """
        Process conversational query with context awareness

        Returns: (result_dict, conversational_response)
        """
        # Detect intent and extract compound
        intent = self.detect_intent(query)
        compound = self.extract_compound(query)

        # Create or update compound context
        if compound:
            if compound not in self.compound_contexts:
                self.compound_contexts[compound] = CompoundContext(
                    smiles=compound,
                    name=None,
                    mol_id=None
                )
            self.current_compound = compound

        # Execute pharmaceutical task based on intent
        result = {}

        try:
            if intent == 'screen':
                if not compound:
                    result = {'status': 'error', 'message': 'No compound specified', 'suggestion': 'Provide SMILES or compound name'}
                else:
                    result = self.corpus.screen_compound(smiles=compound, analysis_level='comprehensive')
                    if compound in self.compound_contexts:
                        self.compound_contexts[compound].analyses_performed.append('screen')
                        self.compound_contexts[compound].results['screen'] = result
                        if 'name' in result:
                            self.compound_contexts[compound].name = result['name']
                        if 'mol_id' in result:
                            self.compound_contexts[compound].mol_id = result['mol_id']

            elif intent == 'dock':
                if not compound:
                    result = {'status': 'error', 'message': 'No compound specified'}
                elif self.tempest_level < 2:
                    result = {'status': 'error', 'message': 'Molecular docking requires TEMPEST Level 2+', 'suggestion': 'Upgrade security clearance'}
                else:
                    result = self.corpus.dock_to_receptors(smiles=compound, receptors=['MOR', 'DOR', 'KOR', 'NMDA'])
                    if compound in self.compound_contexts:
                        self.compound_contexts[compound].analyses_performed.append('dock')
                        self.compound_contexts[compound].results['dock'] = result

            elif intent == 'admet':
                if not compound:
                    result = {'status': 'error', 'message': 'No compound specified'}
                else:
                    result = self.corpus.predict_admet(smiles=compound, use_intel_ai=True, cross_validate=True)
                    if compound in self.compound_contexts:
                        self.compound_contexts[compound].analyses_performed.append('admet')
                        self.compound_contexts[compound].results['admet'] = result

            elif intent == 'bbb':
                if not compound:
                    result = {'status': 'error', 'message': 'No compound specified'}
                else:
                    result = self.corpus.predict_bbb_penetration(smiles=compound, cross_validate=True)
                    if compound in self.compound_contexts:
                        self.compound_contexts[compound].analyses_performed.append('bbb')
                        self.compound_contexts[compound].results['bbb'] = result

            elif intent == 'safety':
                if not compound:
                    result = {'status': 'error', 'message': 'No compound specified'}
                elif self.tempest_level < 2:
                    result = {'status': 'error', 'message': 'Safety profiling requires TEMPEST Level 2+'}
                else:
                    result = self.corpus.comprehensive_safety_profile(compound)
                    if compound in self.compound_contexts:
                        self.compound_contexts[compound].analyses_performed.append('safety')
                        self.compound_contexts[compound].results['safety'] = result

            elif intent == 'abuse':
                if not compound:
                    result = {'status': 'error', 'message': 'No compound specified'}
                elif self.tempest_level < 2:
                    result = {'status': 'error', 'message': 'Abuse potential analysis requires TEMPEST Level 2+'}
                else:
                    comprehensive = self.tempest_level >= 3
                    result = self.corpus.predict_abuse_potential(smiles=compound, comprehensive=comprehensive)
                    if compound in self.compound_contexts:
                        self.compound_contexts[compound].analyses_performed.append('abuse')
                        self.compound_contexts[compound].results['abuse'] = result

            elif intent == 'simulation':
                result = {
                    'status': 'error',
                    'message': 'Patient simulation requires protocol specification',
                    'suggestion': 'Use API or provide full protocol details (compounds, doses, frequencies, duration)'
                }

            elif intent == 'status':
                result = {
                    'status': 'success',
                    'compounds_analyzed': len(self.compound_contexts),
                    'conversation_turns': len(self.conversation_history),
                    'tempest_level': self.tempest_level,
                    'current_compound': self.current_compound,
                    'session_id': self.session_id
                }

            else:
                result = {
                    'status': 'success',
                    'message': f'Intent: {intent}',
                    'suggestion': 'Try: screen, dock, admet, safety, abuse, or status'
                }

        except Exception as e:
            result = {'status': 'error', 'message': str(e)}

        # Build conversational response
        conversational_response = self.build_conversational_response(result, intent, compound)

        # Record conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query=query,
            intent=intent,
            compound=compound,
            action=intent,
            result=result,
            suggestions=self.get_context_suggestions(compound,
                self.compound_contexts.get(compound, CompoundContext(smiles=compound or "", name=None, mol_id=None)).analyses_performed
            ) if compound else [],
            tempest_level=self.tempest_level
        )
        self.conversation_history.append(turn)

        return result, conversational_response

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'tempest_level': self.tempest_level,
            'conversation_turns': len(self.conversation_history),
            'compounds_analyzed': len(self.compound_contexts),
            'current_compound': self.current_compound,
            'compound_contexts': {
                smiles: {
                    'name': ctx.name,
                    'mol_id': ctx.mol_id,
                    'analyses_performed': ctx.analyses_performed,
                    'first_seen': ctx.first_seen,
                    'last_analyzed': ctx.last_analyzed
                }
                for smiles, ctx in self.compound_contexts.items()
            }
        }

    def reset_session(self):
        """Reset conversation session"""
        self.conversation_history.clear()
        self.compound_contexts.clear()
        self.current_compound = None
        self.session_id = f"{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
