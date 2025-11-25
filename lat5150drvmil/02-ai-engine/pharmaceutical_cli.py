#!/usr/bin/env python3
"""
Pharmaceutical Research Corpus - Natural Language CLI
TEMPEST-compliant conversational interface for drug discovery and analysis

Features:
- Natural language compound analysis
- Multi-turn conversational sessions
- Context-aware pharmaceutical research
- Automatic TEMPEST level detection
- Integration with NMDA, NPS, and ZEROPAIN

Usage:
    python3 pharmaceutical_cli.py "screen fentanyl for safety"
    python3 pharmaceutical_cli.py "dock ketamine to NMDA receptor"
    python3 pharmaceutical_cli.py --conversation  # Interactive mode
    python3 pharmaceutical_cli.py --session <file>  # Resume session

Author: LAT5150DRVMIL Integration
Version: 1.0.0
TEMPEST: Class C Compliant
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import pharmaceutical corpus and supporting modules
from sub_agents.pharmaceutical_corpus import PharmaceuticalCorpus, TEMPESTLevel, AuditLogger
from sub_agents.rdkit_wrapper import RDKitAgent
from sub_agents.nmda_agonist_analyzer import NMDAAgonistAnalyzer
from sub_agents.nps_abuse_potential_analyzer import NPSAbusePotentialAnalyzer

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # TEMPEST level colors
    LEVEL_0 = '\033[96m'  # Cyan
    LEVEL_1 = '\033[94m'  # Blue
    LEVEL_2 = '\033[93m'  # Orange/Yellow
    LEVEL_3 = '\033[91m'  # Red


class PharmaceuticalSession:
    """Manage conversational pharmaceutical research sessions"""

    def __init__(self, tempest_level: int = 1, user_id: str = "cli_user", session_file: Optional[str] = None):
        """
        Initialize pharmaceutical session

        Args:
            tempest_level: TEMPEST security level (0-3)
            user_id: User identifier for audit logging
            session_file: Path to load/save session state
        """
        self.tempest_level = tempest_level
        self.user_id = user_id
        self.session_file = session_file

        # Initialize agents
        self.corpus = PharmaceuticalCorpus(tempest_level=tempest_level, user_id=user_id)
        self.rdkit = RDKitAgent()
        self.nmda = NMDAAgonistAnalyzer()
        self.nps = NPSAbusePotentialAnalyzer(verbose=False)

        # Session state
        self.conversation_history: List[Dict[str, Any]] = []
        self.compound_library: Dict[str, Dict[str, Any]] = {}  # id -> compound data
        self.current_compound: Optional[str] = None
        self.session_start = datetime.now()

        # Load session if file provided
        if session_file and Path(session_file).exists():
            self._load_session()

    def _load_session(self):
        """Load session from file"""
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                self.conversation_history = data.get('history', [])
                self.compound_library = data.get('compounds', {})
                self.current_compound = data.get('current_compound')
                print(f"{Colors.OKGREEN}✓ Session loaded: {len(self.conversation_history)} turns{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Failed to load session: {e}{Colors.ENDC}")

    def _save_session(self):
        """Save session to file"""
        if not self.session_file:
            return

        try:
            data = {
                'history': self.conversation_history,
                'compounds': self.compound_library,
                'current_compound': self.current_compound,
                'session_start': self.session_start.isoformat(),
                'tempest_level': self.tempest_level
            }

            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"{Colors.OKGREEN}✓ Session saved to {self.session_file}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Failed to save session: {e}{Colors.ENDC}")

    def parse_natural_language(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured pharmaceutical task

        Returns structured command with:
        - action: screen, dock, admet, safety, simulate, etc.
        - compound: SMILES or name
        - parameters: Additional task parameters
        """
        query_lower = query.lower()

        # Extract SMILES if present (basic pattern)
        smiles_pattern = r'[A-Z][A-Za-z0-9@\+\-\[\]\(\)\=\#]+(?:[A-Za-z0-9@\+\-\[\]\(\)\=\#]*)'
        smiles_match = re.search(smiles_pattern, query)

        # Detect action
        action = 'screen'  # default
        if any(word in query_lower for word in ['dock', 'docking', 'binding affinity']):
            action = 'dock'
        elif any(word in query_lower for word in ['admet', 'pharmacokinetics', 'pk/pd']):
            action = 'admet'
        elif any(word in query_lower for word in ['safety', 'toxicity', 'adverse']):
            action = 'safety'
        elif any(word in query_lower for word in ['abuse', 'nps', 'recreational']):
            action = 'abuse'
        elif any(word in query_lower for word in ['antidepressant', 'nmda', 'ketamine']):
            action = 'nmda'
        elif any(word in query_lower for word in ['simulate', 'patient', 'clinical']):
            action = 'simulate'
        elif any(word in query_lower for word in ['screen', 'analyze', 'test']):
            action = 'screen'

        # Detect compound (SMILES or common names)
        compound = None
        if smiles_match:
            compound = smiles_match.group(0)
        else:
            # Check for common compound names
            compounds_map = {
                'fentanyl': 'CCN(CC)C(=O)C1CN(C)CCc2ccccc21',
                'ketamine': 'CC(=O)C(c1ccccc1Cl)N(C)C',
                'mdma': 'CC(CC1=CC2=C(C=C1)OCO2)NC',
                'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                'cocaine': 'COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C',
                'morphine': 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O',
                'thc': 'CCCCCc1cc(O)c2c(c1)OC(C)(C)C1CCC(C)=CC21'
            }

            for name, smiles in compounds_map.items():
                if name in query_lower:
                    compound = smiles
                    break

        # If no compound found but we have a current compound, use it
        if not compound and self.current_compound:
            compound = self.current_compound

        # Detect receptors for docking
        receptors = []
        if 'mor' in query_lower or 'mu-opioid' in query_lower or 'μ-opioid' in query_lower:
            receptors.append('MOR')
        if 'dor' in query_lower or 'delta-opioid' in query_lower or 'δ-opioid' in query_lower:
            receptors.append('DOR')
        if 'kor' in query_lower or 'kappa-opioid' in query_lower or 'κ-opioid' in query_lower:
            receptors.append('KOR')
        if 'nmda' in query_lower:
            receptors.append('NMDA')
        if '5ht2a' in query_lower or 'serotonin' in query_lower:
            receptors.append('5HT2A')

        # Default receptors for docking
        if action == 'dock' and not receptors:
            receptors = ['MOR', 'DOR', 'KOR']

        # Detect analysis depth
        comprehensive = 'comprehensive' in query_lower or 'detailed' in query_lower or 'full' in query_lower

        return {
            'action': action,
            'compound': compound,
            'receptors': receptors,
            'comprehensive': comprehensive,
            'original_query': query
        }

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parsed pharmaceutical task"""
        action = task['action']
        compound = task['compound']

        if not compound:
            return {
                'status': 'error',
                'message': 'No compound specified. Please provide SMILES or compound name.',
                'suggestion': 'Try: "screen fentanyl" or provide SMILES like "CCN(CC)C(=O)C1CN(C)CCc2ccccc21"'
            }

        try:
            # Execute based on action
            if action == 'screen':
                result = self.corpus.screen_compound(
                    smiles=compound,
                    analysis_level='comprehensive' if task['comprehensive'] else 'standard'
                )

                # Store in library
                if 'mol_id' in result:
                    self.compound_library[result['mol_id']] = result
                    self.current_compound = compound

                return result

            elif action == 'dock':
                result = self.corpus.dock_to_receptors(
                    smiles=compound,
                    receptors=task['receptors']
                )
                return result

            elif action == 'admet':
                result = self.corpus.predict_admet(
                    smiles=compound,
                    use_intel_ai=True,
                    cross_validate=True
                )
                return result

            elif action == 'safety':
                result = self.corpus.comprehensive_safety_profile(compound)
                return result

            elif action == 'abuse':
                result = self.corpus.predict_abuse_potential(
                    smiles=compound,
                    comprehensive=task['comprehensive']
                )
                return result

            elif action == 'nmda':
                # Parse with RDKit first
                rdkit_result = self.rdkit.parse_smiles(smiles=compound, mol_id='temp_nmda')
                if rdkit_result['status'] == 'success':
                    mol_id = rdkit_result['mol_id']
                    result = self.nmda.comprehensive_analysis(mol_id=mol_id)
                    return result
                else:
                    return rdkit_result

            elif action == 'simulate':
                return {
                    'status': 'error',
                    'message': 'Patient simulation requires protocol specification',
                    'suggestion': 'Use API for patient simulation with full protocol details'
                }

            else:
                return {
                    'status': 'error',
                    'message': f'Unknown action: {action}',
                    'suggestion': 'Try: screen, dock, admet, safety, abuse, nmda'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'action': action
            }

    def format_result(self, result: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Format result for terminal display"""
        if result.get('status') == 'error':
            return f"{Colors.FAIL}✗ Error: {result.get('message', 'Unknown error')}{Colors.ENDC}\n"

        output = []
        action = task['action']

        # Header
        tempest_color = [Colors.LEVEL_0, Colors.LEVEL_1, Colors.LEVEL_2, Colors.LEVEL_3][self.tempest_level]
        output.append(f"\n{tempest_color}{'=' * 70}{Colors.ENDC}")
        output.append(f"{Colors.BOLD}Pharmaceutical Analysis: {action.upper()}{Colors.ENDC}")
        output.append(f"{tempest_color}{'=' * 70}{Colors.ENDC}\n")

        # Action-specific formatting
        if action == 'screen':
            output.append(f"{Colors.OKBLUE}Compound:{Colors.ENDC} {result.get('name', 'Unknown')}")
            output.append(f"{Colors.OKBLUE}SMILES:{Colors.ENDC} {result.get('smiles', 'N/A')}")

            if 'properties' in result:
                props = result['properties']
                output.append(f"\n{Colors.BOLD}Molecular Properties:{Colors.ENDC}")
                output.append(f"  MW: {props.get('molecular_weight', 'N/A'):.2f} g/mol")
                output.append(f"  LogP: {props.get('logp', 'N/A'):.2f}")
                output.append(f"  TPSA: {props.get('tpsa', 'N/A'):.2f} Ų")

            if 'drug_likeness' in result:
                dl = result['drug_likeness']
                output.append(f"\n{Colors.BOLD}Drug-Likeness:{Colors.ENDC}")
                output.append(f"  Lipinski: {Colors.OKGREEN if dl.get('lipinski_pass') else Colors.FAIL}{'PASS' if dl.get('lipinski_pass') else 'FAIL'}{Colors.ENDC}")
                output.append(f"  QED Score: {dl.get('qed_score', 'N/A'):.2f}")

            if 'nps_classification' in result:
                nps = result['nps_classification']
                output.append(f"\n{Colors.BOLD}NPS Classification:{Colors.ENDC}")
                output.append(f"  Class: {nps.get('class', 'Unknown')}")
                if nps.get('is_nps'):
                    output.append(f"  {Colors.WARNING}⚠ Novel Psychoactive Substance detected{Colors.ENDC}")

        elif action == 'dock':
            output.append(f"\n{Colors.BOLD}Docking Results:{Colors.ENDC}")
            if 'docking_results' in result:
                for receptor, data in result['docking_results'].items():
                    affinity = data.get('binding_affinity', 'N/A')
                    ki = data.get('ki_nm', 'N/A')
                    output.append(f"  {receptor}: {affinity} kcal/mol (Ki: {ki} nM)")

        elif action == 'admet':
            output.append(f"\n{Colors.BOLD}ADMET Predictions:{Colors.ENDC}")
            if 'bbb_permeability' in result:
                output.append(f"  BBB Penetration: {result.get('bbb_permeability', 'N/A')}")
            if 'bioavailability' in result:
                output.append(f"  Oral Bioavailability: {result.get('bioavailability', 'N/A'):.1f}%")
            if 'toxicity' in result:
                tox = result['toxicity']
                output.append(f"\n{Colors.BOLD}Toxicity:{Colors.ENDC}")
                for key, value in tox.items():
                    output.append(f"  {key}: {value}")

        elif action == 'safety':
            if 'safety_score' in result:
                score = result['safety_score']
                color = Colors.OKGREEN if score >= 7 else Colors.WARNING if score >= 4 else Colors.FAIL
                output.append(f"\n{Colors.BOLD}Safety Score:{Colors.ENDC} {color}{score:.1f}/10{Colors.ENDC}")

            if 'warnings' in result:
                output.append(f"\n{Colors.WARNING}Warnings:{Colors.ENDC}")
                for warning in result['warnings']:
                    output.append(f"  ⚠ {warning}")

        elif action == 'abuse':
            if 'abuse_score' in result:
                score = result['abuse_score']
                color = Colors.FAIL if score >= 7 else Colors.WARNING if score >= 4 else Colors.OKGREEN
                output.append(f"\n{Colors.BOLD}Abuse Potential:{Colors.ENDC} {color}{score:.1f}/10{Colors.ENDC}")

            if 'nps_class' in result:
                output.append(f"\n{Colors.BOLD}Classification:{Colors.ENDC} {result['nps_class']}")

        # Footer
        output.append(f"\n{tempest_color}{'=' * 70}{Colors.ENDC}\n")

        return '\n'.join(output)

    def process_query(self, query: str) -> str:
        """Process natural language query and return formatted result"""
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'tempest_level': self.tempest_level
        })

        # Parse query
        task = self.parse_natural_language(query)

        # Execute task
        result = self.execute_task(task)

        # Store result in history
        self.conversation_history[-1]['task'] = task
        self.conversation_history[-1]['result'] = result

        # Format and return
        formatted = self.format_result(result, task)

        # Auto-save session if file specified
        if self.session_file:
            self._save_session()

        return formatted

    def interactive_mode(self):
        """Run interactive conversational session"""
        print(f"\n{Colors.HEADER}{'=' * 70}")
        print(f"  Pharmaceutical Research Corpus - Interactive Mode")
        print(f"  TEMPEST Level: {self.tempest_level}")
        print(f"  Type 'help' for commands, 'exit' to quit")
        print(f"{'=' * 70}{Colors.ENDC}\n")

        while True:
            try:
                # Prompt
                tempest_color = [Colors.LEVEL_0, Colors.LEVEL_1, Colors.LEVEL_2, Colors.LEVEL_3][self.tempest_level]
                query = input(f"{tempest_color}pharma>{Colors.ENDC} ")

                if not query.strip():
                    continue

                query = query.strip()

                # Handle special commands
                if query.lower() in ['exit', 'quit', 'q']:
                    print(f"\n{Colors.OKGREEN}Session complete. {len(self.conversation_history)} turns.{Colors.ENDC}")
                    if self.session_file:
                        self._save_session()
                    break

                elif query.lower() == 'help':
                    self._print_help()
                    continue

                elif query.lower() == 'status':
                    self._print_status()
                    continue

                elif query.lower() == 'library':
                    self._print_library()
                    continue

                elif query.lower().startswith('level '):
                    try:
                        new_level = int(query.split()[1])
                        if 0 <= new_level <= 3:
                            self.tempest_level = new_level
                            self.corpus.tempest_level = new_level
                            print(f"{Colors.OKGREEN}✓ TEMPEST level set to {new_level}{Colors.ENDC}")
                        else:
                            print(f"{Colors.FAIL}✗ Level must be 0-3{Colors.ENDC}")
                    except:
                        print(f"{Colors.FAIL}✗ Invalid level command{Colors.ENDC}")
                    continue

                # Process pharmaceutical query
                result = self.process_query(query)
                print(result)

            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}Interrupted. Type 'exit' to quit.{Colors.ENDC}\n")
            except Exception as e:
                print(f"\n{Colors.FAIL}✗ Error: {e}{Colors.ENDC}\n")

    def _print_help(self):
        """Print help message"""
        print(f"\n{Colors.BOLD}Pharmaceutical Research CLI - Commands:{Colors.ENDC}")
        print(f"\n{Colors.OKBLUE}Analysis Commands:{Colors.ENDC}")
        print(f"  screen <compound>         - Comprehensive compound screening")
        print(f"  dock <compound> to <rec>  - Molecular docking")
        print(f"  admet <compound>          - ADMET prediction")
        print(f"  safety <compound>         - Safety profile")
        print(f"  abuse <compound>          - Abuse potential")
        print(f"\n{Colors.OKBLUE}Session Commands:{Colors.ENDC}")
        print(f"  status                    - Show session status")
        print(f"  library                   - Show compound library")
        print(f"  level <0-3>               - Set TEMPEST level")
        print(f"  help                      - Show this help")
        print(f"  exit                      - Exit session\n")

    def _print_status(self):
        """Print session status"""
        elapsed = datetime.now() - self.session_start
        print(f"\n{Colors.BOLD}Session Status:{Colors.ENDC}")
        print(f"  TEMPEST Level: {self.tempest_level}")
        print(f"  User: {self.user_id}")
        print(f"  Conversation Turns: {len(self.conversation_history)}")
        print(f"  Compounds Analyzed: {len(self.compound_library)}")
        print(f"  Session Duration: {elapsed}")
        if self.current_compound:
            print(f"  Current Compound: {self.current_compound}")
        print()

    def _print_library(self):
        """Print compound library"""
        if not self.compound_library:
            print(f"\n{Colors.WARNING}No compounds in library yet.{Colors.ENDC}\n")
            return

        print(f"\n{Colors.BOLD}Compound Library ({len(self.compound_library)}):{Colors.ENDC}\n")
        for i, (mol_id, data) in enumerate(self.compound_library.items(), 1):
            name = data.get('name', 'Unknown')
            smiles = data.get('smiles', 'N/A')[:50]
            print(f"  {i}. {name} ({mol_id})")
            print(f"     {smiles}...")
        print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Pharmaceutical Research Corpus - Natural Language CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pharmaceutical_cli.py "screen fentanyl for safety"
  pharmaceutical_cli.py "dock ketamine to NMDA receptor"
  pharmaceutical_cli.py --conversation --level 2
  pharmaceutical_cli.py --session pharma.json --conversation
        """
    )

    parser.add_argument('query', nargs='?', help='Natural language query')
    parser.add_argument('-c', '--conversation', action='store_true', help='Interactive conversation mode')
    parser.add_argument('-l', '--level', type=int, default=1, choices=[0, 1, 2, 3], help='TEMPEST security level (0-3)')
    parser.add_argument('-s', '--session', help='Session file to load/save')
    parser.add_argument('-u', '--user', default='cli_user', help='User ID for audit logging')

    args = parser.parse_args()

    # Initialize session
    session = PharmaceuticalSession(
        tempest_level=args.level,
        user_id=args.user,
        session_file=args.session
    )

    # Interactive mode
    if args.conversation or not args.query:
        session.interactive_mode()
    else:
        # Single query mode
        result = session.process_query(args.query)
        print(result)

        # Save session if file specified
        if args.session:
            session._save_session()


if __name__ == '__main__':
    main()
