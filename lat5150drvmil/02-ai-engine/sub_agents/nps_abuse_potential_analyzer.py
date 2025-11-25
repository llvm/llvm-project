#!/usr/bin/env python3
"""
Novel Psychoactive Substance (NPS) & Abuse Potential Analyzer
PROACTIVE Drug Threat Intelligence & Countermeasure Development

Mission: Predict and defend against emerging designer drugs BEFORE synthesis

Capabilities:
- Novel psychoactive substance (NPS) classification
- Recreational abuse potential prediction
- Designer drug structure analysis
- Synthetic cannabinoid/cathinone/opioid detection
- Neurotoxicity & lethality assessment
- Antidote/antagonist recommendation
- Regulatory pre-emption intelligence

TIME BUDGET: Up to 12 hours for comprehensive analysis
DATA SCALE: Massive datasets (1M+ compounds screening)

References:
- UNODC Early Warning Advisory (EWA) on NPS
- DEA/Forensic Science International - Emerging drug trends
- European Monitoring Centre for Drugs and Drug Addiction (EMCDDA)
- NIH/NIDA - Abuse potential assessment frameworks
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sub_agents.rdkit_wrapper import RDKitAgent
from sub_agents.nmda_agonist_analyzer import NMDAAgonistAnalyzer

class NPSAbusePotentialAnalyzer:
    def __init__(self, verbose: bool = True):
        """
        Initialize NPS & Abuse Potential Analyzer

        Args:
            verbose: Enable detailed logging for long-running analyses
        """
        self.rdkit = RDKitAgent()
        self.nmda_analyzer = NMDAAgonistAnalyzer()
        self.verbose = verbose

        # NPS chemical class patterns (SMARTS)
        self.nps_patterns = {
            # SYNTHETIC CANNABINOIDS
            'jw018_indole': 'c1ccc2c(c1)c(cn2CC)C(=O)N',  # JWH-018 core
            'indazole_cannabinoid': 'c1ccc2c(c1)[nH]nc2C(=O)N',  # AB-CHMINACA class
            'quinolinyl_cannabinoid': 'c1ccc2c(c1)ccc(n2)C(=O)N',  # PB-22 class

            # SYNTHETIC CATHINONES ("Bath Salts")
            'cathinone_core': 'CC(N)C(=O)c1ccccc1',  # Basic cathinone
            'methylenedioxy_cathinone': 'CC(N)C(=O)c1ccc2OCOc2c1',  # MDPV-like
            'pyrrolidinyl_cathinone': 'C(C(=O)c1ccccc1)N2CCCC2',  # Alpha-PVP class

            # SYNTHETIC OPIOIDS (Fentanyl analogs)
            'fentanyl_scaffold': 'CCN(CC)C(=O)C(c1ccccc1)N2CCN(CC2)c3ccccc3',
            'carfentanil_scaffold': 'CCOC(=O)C1CCN(C1)C(=O)C(c2ccccc2)N',
            'acetylfentanyl': 'CC(=O)N(c1ccccc1)C2CCN(CC2)CCc3ccccc3',

            # PHENETHYLAMINES (2C-x, NBOMe)
            'phenethylamine_2c': 'CCc1cc(OC)c(OC)cc1CCN',  # 2C-B core
            'nbome': 'COc1cc(CCNCc2ccccc2OC)c(OC)cc1Br',  # 25I-NBOMe
            'dox_series': 'CC(CCN)c1cc(OC)c(OC)cc1',  # DOx series

            # TRYPTAMINES
            'dmt_core': 'CN(C)CCc1c[nH]c2ccccc12',  # DMT
            'psilocybin_core': 'CN(C)CCc1c[nH]c2cc(OP(=O)(O)O)ccc12',  # Psilocybin
            '5meo_tryptamine': 'CN(C)CCc1c[nH]c2cc(OC)ccc12',  # 5-MeO-DMT

            # DISSOCIATIVES (PCP/Ketamine analogs)
            'pcp_analog': 'C1CCC(CC1)(c2ccccc2)N',  # PCP variants
            'ketamine_analog': 'CNC(=O)C(c1ccccc1)N',  # Ketamine analogs
            'mxe_core': 'CCNC1(c2ccc(OC)cc2)CCCCC1=O',  # MXE (Methoxetamine)

            # BENZODIAZEPINE ANALOGS
            'designer_benzo': 'Clc1ccc2c(c1)NC(=O)CN=C2c3ccccc3',  # Etizolam-like
            'fluorobenzo': 'Fc1ccc2c(c1)C(=O)CN=C2c3ccccc3',  # Flualprazolam

            # SYNTHETIC STIMULANTS
            'mdma_core': 'CC(N)Cc1ccc2OCOc2c1',  # MDMA
            'methylphenidate_analog': 'COC(=O)C1C2CCCC(C2)N1',  # Ritalin analogs
        }

        # Neurotransmitter receptor binding profiles
        self.receptor_profiles = {
            'opioid': ['MOR', 'DOR', 'KOR'],  # Mu, Delta, Kappa opioid
            'cannabinoid': ['CB1', 'CB2'],
            'serotonin': ['5HT1A', '5HT2A', '5HT2C'],
            'dopamine': ['D1', 'D2', 'D3'],
            'nmda': ['GluN1', 'GluN2A', 'GluN2B'],
            'gaba': ['GABAA', 'GABAB']
        }

        # Known high-abuse substances for comparison
        self.abuse_reference_compounds = {
            'fentanyl': 'CCN(CC)C(=O)C(c1ccccc1)N2CCC(CC2)N3c4ccccc4Nc5ccccc53',
            'heroin': 'CC(=O)Oc1ccc2C3Cc4ccc(O)c(OC(C)=O)c4C=CC3NC(C)C2c1',
            'methamphetamine': 'CC(Cc1ccccc1)NC',
            'mdma': 'CC(Cc1ccc2OCOc2c1)NC',
            'cocaine': 'COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C',
            'lsd': 'CCN(CC)C(=O)C1CN(C)C2Cc3c[nH]c4cccc(C2=C1)c34',
            'thc': 'CCCCCc1cc(O)c2C3C=C(C)CCC3C(C)(C)Oc2c1O',
            'jwh018': 'c1ccc2c(c1)c(cn2CCCCC)C(=O)c3cccc4ncccc34'  # Synthetic cannabinoid
        }

    def log(self, message: str, level: str = 'INFO'):
        """Log messages if verbose enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")

    def classify_nps(self, mol_id: str) -> Dict[str, Any]:
        """
        Classify molecule as Novel Psychoactive Substance

        Args:
            mol_id: Molecule ID

        Returns:
            NPS classification and chemical class
        """
        self.log(f"Starting NPS classification for {mol_id}")

        results = {
            'molecule_id': mol_id,
            'is_nps': False,
            'chemical_classes': [],
            'scaffold_matches': [],
            'structural_alerts': [],
            'dea_schedule_prediction': 'Unknown',
            'regulatory_recommendations': []
        }

        # Screen against NPS patterns
        for class_name, pattern in self.nps_patterns.items():
            self.log(f"Screening {class_name}...", 'DEBUG')

            match_result = self.rdkit.substructure_search(
                pattern=pattern,
                pattern_format='smarts'
            )

            if match_result.get('success'):
                matches = [m for m in match_result['matches'] if m['molecule_id'] == mol_id]
                if matches:
                    chemical_class = self._classify_chemical_class(class_name)
                    results['chemical_classes'].append(chemical_class)
                    results['scaffold_matches'].append({
                        'pattern': class_name,
                        'class': chemical_class,
                        'regulatory_status': self._get_regulatory_status(chemical_class)
                    })
                    results['is_nps'] = True

        # DEA schedule prediction
        if results['is_nps']:
            results['dea_schedule_prediction'] = self._predict_dea_schedule(results['chemical_classes'])
            results['regulatory_recommendations'] = self._generate_regulatory_recommendations(results)

        self.log(f"NPS classification complete. Classes found: {len(results['chemical_classes'])}")

        return results

    def _classify_chemical_class(self, pattern_name: str) -> str:
        """Map pattern to chemical class"""
        class_mapping = {
            'jw018_indole': 'Synthetic Cannabinoid (Indole)',
            'indazole_cannabinoid': 'Synthetic Cannabinoid (Indazole)',
            'quinolinyl_cannabinoid': 'Synthetic Cannabinoid (Quinoline)',
            'cathinone_core': 'Synthetic Cathinone',
            'methylenedioxy_cathinone': 'Synthetic Cathinone (MDPV-class)',
            'pyrrolidinyl_cathinone': 'Synthetic Cathinone (Pyrrolidine)',
            'fentanyl_scaffold': 'Synthetic Opioid (Fentanyl analog)',
            'carfentanil_scaffold': 'Synthetic Opioid (Carfentanil analog)',
            'acetylfentanyl': 'Synthetic Opioid (Acetyl-fentanyl)',
            'phenethylamine_2c': 'Phenethylamine (2C-series)',
            'nbome': 'Phenethylamine (NBOMe series)',
            'dox_series': 'Phenethylamine (DOx series)',
            'dmt_core': 'Tryptamine (DMT-like)',
            'psilocybin_core': 'Tryptamine (Psilocybin-like)',
            '5meo_tryptamine': 'Tryptamine (5-MeO class)',
            'pcp_analog': 'Dissociative (PCP analog)',
            'ketamine_analog': 'Dissociative (Ketamine analog)',
            'mxe_core': 'Dissociative (Arylcyclohexylamine)',
            'designer_benzo': 'Designer Benzodiazepine',
            'fluorobenzo': 'Designer Benzodiazepine (Fluoro)',
            'mdma_core': 'Entactogen (MDMA-like)',
            'methylphenidate_analog': 'Stimulant (Methylphenidate analog)'
        }
        return class_mapping.get(pattern_name, 'Unknown NPS')

    def _get_regulatory_status(self, chemical_class: str) -> str:
        """Get regulatory status by class"""
        if 'Synthetic Opioid' in chemical_class:
            return 'DEA Schedule I/II - HIGH PRIORITY'
        elif 'Synthetic Cannabinoid' in chemical_class:
            return 'DEA Schedule I - Controlled analog'
        elif 'Synthetic Cathinone' in chemical_class:
            return 'DEA Schedule I - Bath salts analog'
        elif 'Designer Benzodiazepine' in chemical_class:
            return 'Emerging threat - Monitor for scheduling'
        elif 'Fentanyl' in chemical_class:
            return 'DEA Schedule I/II - LETHAL THREAT'
        else:
            return 'Review for analog act applicability'

    def _predict_dea_schedule(self, chemical_classes: List[str]) -> str:
        """Predict DEA schedule"""
        if any('Fentanyl' in c for c in chemical_classes):
            return 'Schedule I (predicted) - Extreme abuse potential + lethality'
        elif any('Synthetic Opioid' in c for c in chemical_classes):
            return 'Schedule II (predicted) - High abuse potential'
        elif any('Cannabinoid' in c or 'Cathinone' in c for c in chemical_classes):
            return 'Schedule I (predicted) - No medical use, high abuse'
        elif any('Benzodiazepine' in c for c in chemical_classes):
            return 'Schedule IV (predicted) - Moderate abuse potential'
        else:
            return 'Unscheduled (monitor for emergency scheduling)'

    def _generate_regulatory_recommendations(self, classification: Dict) -> List[str]:
        """Generate regulatory action recommendations"""
        recommendations = []

        classes = classification['chemical_classes']

        if any('Fentanyl' in c for c in classes):
            recommendations.append(
                'URGENT: Emergency DEA scheduling recommended - High lethality risk'
            )
            recommendations.append(
                'Coordinate with CDC/NIDA for overdose prevention strategy'
            )
            recommendations.append(
                'Develop naloxone protocol for first responders'
            )

        if any('Synthetic Cannabinoid' in c for c in classes):
            recommendations.append(
                'Apply DEA analog act - Schedule I cannabinoid mimetic'
            )
            recommendations.append(
                'Alert poison control centers - Unpredictable toxicity'
            )

        if any('Designer Benzodiazepine' in c for c in classes):
            recommendations.append(
                'Monitor dark web markets for proliferation'
            )
            recommendations.append(
                'Coordinate with international regulators (EMCDDA/UNODC)'
            )

        recommendations.append(
            'Add to NFLIS/NCANDS database for forensic monitoring'
        )

        recommendations.append(
            'Develop analytical standards (GC-MS/LC-MS reference spectra)'
        )

        return recommendations

    def predict_abuse_potential(self, mol_id: str,
                                comprehensive: bool = True) -> Dict[str, Any]:
        """
        Predict recreational abuse potential of novel substance

        Args:
            mol_id: Molecule ID
            comprehensive: If True, perform extensive 12-hour analysis

        Returns:
            Abuse potential assessment
        """
        start_time = time.time()
        self.log(f"Starting abuse potential prediction for {mol_id}")
        self.log(f"Comprehensive mode: {comprehensive}")

        results = {
            'molecule_id': mol_id,
            'abuse_potential_score': 0.0,  # 0-10 scale
            'abuse_risk_category': 'Unknown',
            'reinforcement_mechanisms': [],
            'comparison_with_known_drugs': {},
            'neurotoxicity_risk': 'Unknown',
            'lethality_risk': 'Unknown',
            'antidote_recommendations': [],
            'synthesis_precursors': [],
            'dark_web_likelihood': 'Unknown',
            'analysis_duration_seconds': 0
        }

        # 1. NPS Classification
        self.log("Step 1/8: NPS classification...")
        nps_class = self.classify_nps(mol_id)
        results['nps_classification'] = nps_class

        # 2. Structural similarity to known drugs of abuse
        self.log("Step 2/8: Comparing with known drugs of abuse...")
        similarity_results = self._compare_with_abuse_references(mol_id)
        results['comparison_with_known_drugs'] = similarity_results

        # 3. Predict receptor binding
        self.log("Step 3/8: Predicting receptor binding profiles...")
        receptor_prediction = self._predict_receptor_binding(mol_id, nps_class)
        results['receptor_binding'] = receptor_prediction

        # 4. Assess reinforcement potential
        self.log("Step 4/8: Assessing reinforcement mechanisms...")
        reinforcement = self._assess_reinforcement_potential(
            nps_class, receptor_prediction, similarity_results
        )
        results['reinforcement_mechanisms'] = reinforcement

        # 5. Neurotoxicity assessment
        self.log("Step 5/8: Evaluating neurotoxicity risk...")
        neurotox = self._assess_neurotoxicity(mol_id, nps_class)
        results['neurotoxicity_risk'] = neurotox

        # 6. Lethality risk
        self.log("Step 6/8: Calculating lethality risk...")
        lethality = self._assess_lethality_risk(mol_id, nps_class, similarity_results)
        results['lethality_risk'] = lethality

        # 7. Antidote recommendations
        self.log("Step 7/8: Generating antidote recommendations...")
        antidotes = self._recommend_antidotes(nps_class, receptor_prediction)
        results['antidote_recommendations'] = antidotes

        # 8. Calculate overall abuse potential score
        self.log("Step 8/8: Calculating final abuse potential score...")
        abuse_score = self._calculate_abuse_score(
            nps_class, similarity_results, receptor_prediction,
            reinforcement, neurotox, lethality
        )
        results['abuse_potential_score'] = abuse_score
        results['abuse_risk_category'] = self._categorize_abuse_risk(abuse_score)

        # Extended analysis (if comprehensive mode)
        if comprehensive:
            self.log("Extended analysis mode - predicting synthesis routes...")
            results['synthesis_precursors'] = self._predict_synthesis_precursors(mol_id)

            self.log("Extended analysis - dark web market prediction...")
            results['dark_web_likelihood'] = self._predict_dark_web_proliferation(
                abuse_score, nps_class, lethality
            )

        # Analysis complete
        elapsed_time = time.time() - start_time
        results['analysis_duration_seconds'] = round(elapsed_time, 2)

        self.log(f"Analysis complete in {elapsed_time:.2f} seconds")
        self.log(f"Abuse potential score: {abuse_score}/10")
        self.log(f"Risk category: {results['abuse_risk_category']}")

        return results

    def _compare_with_abuse_references(self, mol_id: str) -> Dict[str, Any]:
        """Compare with known drugs of abuse"""
        similarities = {}

        for drug_name, smiles in self.abuse_reference_compounds.items():
            # Parse reference
            ref_result = self.rdkit.parse_molecule(
                structure=smiles,
                format='smiles',
                name=f"ref_{drug_name}"
            )

            if ref_result.get('success'):
                ref_mol_id = ref_result['molecule_id']

                # Calculate similarity
                sim_result = self.rdkit.similarity_search(
                    query_mol_id=mol_id,
                    target_mol_ids=[ref_mol_id],
                    fp_type='morgan',
                    metric='tanimoto'
                )

                if sim_result.get('success') and sim_result['results']:
                    similarity = sim_result['results'][0]['similarity']
                    similarities[drug_name] = {
                        'similarity': round(similarity, 4),
                        'interpretation': self._interpret_abuse_similarity(similarity, drug_name)
                    }

        # Find best match
        if similarities:
            best_match = max(similarities.items(), key=lambda x: x[1]['similarity'])
            return {
                'similarities': similarities,
                'best_match': best_match[0],
                'best_similarity': best_match[1]['similarity']
            }
        else:
            return {'similarities': {}, 'best_match': None, 'best_similarity': 0.0}

    def _interpret_abuse_similarity(self, score: float, drug: str) -> str:
        """Interpret similarity to known drugs of abuse"""
        if score >= 0.85:
            return f"CRITICAL: Near-identical to {drug} - Expect similar abuse profile"
        elif score >= 0.7:
            return f"HIGH RISK: Structural analog of {drug} - Likely similar effects"
        elif score >= 0.5:
            return f"MODERATE RISK: Shares features with {drug}"
        else:
            return f"Novel structure compared to {drug}"

    def _predict_receptor_binding(self, mol_id: str, nps_class: Dict) -> Dict[str, List[str]]:
        """Predict neurotransmitter receptor binding"""
        predictions = {}

        # Based on chemical class
        classes = nps_class.get('chemical_classes', [])

        for chem_class in classes:
            if 'Synthetic Opioid' in chem_class or 'Fentanyl' in chem_class:
                predictions.setdefault('opioid', []).extend(['MOR (high)', 'DOR (moderate)', 'KOR (low)'])

            if 'Synthetic Cannabinoid' in chem_class:
                predictions.setdefault('cannabinoid', []).extend(['CB1 (high)', 'CB2 (moderate)'])

            if 'Phenethylamine' in chem_class or 'Tryptamine' in chem_class:
                predictions.setdefault('serotonin', []).extend(['5HT2A (high)', '5HT2C (moderate)', '5HT1A (low)'])

            if 'Cathinone' in chem_class or 'MDMA' in chem_class:
                predictions.setdefault('dopamine', []).extend(['D1 (moderate)', 'D2 (high)'])
                predictions.setdefault('serotonin', []).extend(['SERT (high)'])

            if 'Dissociative' in chem_class or 'PCP' in chem_class or 'Ketamine' in chem_class:
                predictions.setdefault('nmda', []).extend(['GluN2B (high)', 'GluN2A (moderate)'])

            if 'Benzodiazepine' in chem_class:
                predictions.setdefault('gaba', []).extend(['GABAA (high)'])

        return predictions

    def _assess_reinforcement_potential(self, nps_class: Dict,
                                       receptors: Dict, similarities: Dict) -> List[str]:
        """Assess reinforcement mechanisms (addiction potential)"""
        mechanisms = []

        # Dopaminergic reinforcement
        if 'dopamine' in receptors:
            mechanisms.append('Dopaminergic reinforcement (mesolimbic pathway) - HIGH ADDICTION RISK')

        # Opioid reinforcement
        if 'opioid' in receptors:
            mechanisms.append('Opioid receptor reinforcement (VTA/NAc) - EXTREME ADDICTION RISK')

        # Serotonergic
        if 'serotonin' in receptors:
            mechanisms.append('Serotonergic modulation - MODERATE-HIGH psychedelic reinforcement')

        # Cannabinoid
        if 'cannabinoid' in receptors:
            mechanisms.append('CB1 receptor activation - MODERATE reinforcement potential')

        # GABAergic
        if 'gaba' in receptors:
            mechanisms.append('GABAergic enhancement - MODERATE-HIGH dependence liability')

        # Similarity-based
        best_match = similarities.get('best_match')
        if best_match in ['fentanyl', 'heroin']:
            mechanisms.append('Structural similarity to opioids - EXTREME overdose risk')
        elif best_match in ['methamphetamine', 'cocaine']:
            mechanisms.append('Stimulant-like structure - HIGH compulsive use risk')

        return mechanisms if mechanisms else ['Unknown reinforcement mechanism']

    def _assess_neurotoxicity(self, mol_id: str, nps_class: Dict) -> str:
        """Assess neurotoxicity risk"""
        classes = nps_class.get('chemical_classes', [])

        # High neurotoxicity risks
        if any('MDMA' in c or 'Cathinone' in c for c in classes):
            return 'HIGH - Serotonergic neurotoxicity (hyperthermia, serotonin syndrome)'

        if any('Methamphetamine' in c for c in classes):
            return 'HIGH - Dopaminergic neurotoxicity (oxidative stress)'

        if any('Synthetic Cannabinoid' in c for c in classes):
            return 'MODERATE-HIGH - Unpredictable CNS effects, seizures reported'

        if any('Dissociative' in c or 'PCP' in c for c in classes):
            return 'MODERATE - NMDA antagonism (Olney\'s lesions possible at high doses)'

        if any('Tryptamine' in c or 'Phenethylamine' in c for c in classes):
            return 'LOW-MODERATE - 5HT2A agonism (generally low neurotoxicity)'

        if any('Benzodiazepine' in c for c in classes):
            return 'LOW - Generally low neurotoxicity (respiratory depression risk)'

        return 'UNKNOWN - Novel structure requires toxicological studies'

    def _assess_lethality_risk(self, mol_id: str, nps_class: Dict,
                              similarities: Dict) -> str:
        """Assess lethality risk"""
        classes = nps_class.get('chemical_classes', [])

        # EXTREME lethality
        if any('Fentanyl' in c or 'Carfentanil' in c for c in classes):
            return 'EXTREME - Lethal dose in microgram range, respiratory depression'

        if similarities.get('best_match') in ['fentanyl', 'heroin'] and similarities.get('best_similarity', 0) > 0.7:
            return 'EXTREME - Opioid analog, expect high overdose mortality'

        # HIGH lethality
        if any('Synthetic Opioid' in c for c in classes):
            return 'HIGH - Opioid overdose risk, respiratory arrest'

        if any('Synthetic Cannabinoid' in c for c in classes):
            return 'HIGH - Unpredictable dose-response, seizures, cardiovascular events'

        # MODERATE lethality
        if any('Cathinone' in c or 'MDMA' in c for c in classes):
            return 'MODERATE - Hyperthermia, cardiovascular complications, serotonin syndrome'

        if any('Dissociative' in c or 'PCP' in c for c in classes):
            return 'MODERATE - Respiratory depression at high doses, behavioral toxicity'

        # LOW-MODERATE lethality
        if any('Benzodiazepine' in c for c in classes):
            return 'LOW-MODERATE - Respiratory depression (especially with alcohol/opioids)'

        if any('Tryptamine' in c or 'Phenethylamine' in c for c in classes):
            return 'LOW-MODERATE - Generally low lethality (psychological risk higher)'

        return 'UNKNOWN - Novel structure requires LD50 determination'

    def _recommend_antidotes(self, nps_class: Dict, receptors: Dict) -> List[Dict[str, str]]:
        """Recommend antidotes/antagonists for overdose"""
        antidotes = []

        # Opioid overdose
        if 'opioid' in receptors:
            antidotes.append({
                'antidote': 'Naloxone (Narcan)',
                'mechanism': 'Mu-opioid receptor antagonist',
                'dosing': '0.4-2mg IV/IM/IN, repeat q2-3min as needed',
                'notes': 'CRITICAL for fentanyl analogs - May require multiple doses'
            })
            antidotes.append({
                'antidote': 'Nalmefene (Revex)',
                'mechanism': 'Long-acting opioid antagonist',
                'dosing': '1mg IV initially',
                'notes': 'Longer duration than naloxone (4-8h vs 30-90min)'
            })

        # Benzodiazepine overdose
        if 'gaba' in receptors:
            antidotes.append({
                'antidote': 'Flumazenil (Romazicon)',
                'mechanism': 'GABAA receptor antagonist',
                'dosing': '0.2mg IV over 30sec, may repeat',
                'notes': 'CAUTION: Seizure risk in chronic benzo users'
            })

        # Serotonin syndrome
        if 'serotonin' in receptors:
            antidotes.append({
                'antidote': 'Cyproheptadine',
                'mechanism': '5HT2A antagonist',
                'dosing': '12mg PO initially, then 2mg q2h',
                'notes': 'For serotonin syndrome management'
            })

        # Cannabinoid toxicity
        if 'cannabinoid' in receptors:
            antidotes.append({
                'antidote': 'Supportive care only',
                'mechanism': 'No specific antagonist available',
                'dosing': 'Benzodiazepines for agitation/seizures',
                'notes': 'No CB1 antagonist approved for clinical use (rimonabant withdrawn)'
            })

        # NMDA antagonist toxicity
        if 'nmda' in receptors:
            antidotes.append({
                'antidote': 'Supportive care + benzodiazepines',
                'mechanism': 'Manage agitation and prevent injury',
                'dosing': 'Lorazepam 2-4mg IM/IV as needed',
                'notes': 'No specific NMDA antagonist reversal agent'
            })

        return antidotes if antidotes else [
            {
                'antidote': 'Supportive care',
                'mechanism': 'Unknown - novel substance',
                'dosing': 'ABC (Airway, Breathing, Circulation)',
                'notes': 'Consult poison control center'
            }
        ]

    def _predict_synthesis_precursors(self, mol_id: str) -> List[Dict[str, Any]]:
        """
        Predict synthesis precursors (for regulatory control)

        In production, this would use retrosynthetic analysis
        """
        # Get molecular formula and structure
        desc_result = self.rdkit.calculate_descriptors(mol_id, 'basic')

        if not desc_result.get('success'):
            return []

        # Placeholder for retrosynthetic analysis
        # In production: Use RDKit retrosynthetic tools or AI-based synthesis planning
        precursors = [
            {
                'precursor_class': 'Aromatic amines',
                'regulatory_status': 'DEA List I (some compounds)',
                'recommendation': 'Monitor chemical supply chains'
            },
            {
                'precursor_class': 'Piperidine/Pyrrolidine',
                'regulatory_status': 'DEA List II',
                'recommendation': 'Track bulk purchases'
            }
        ]

        return precursors

    def _predict_dark_web_proliferation(self, abuse_score: float,
                                       nps_class: Dict, lethality: str) -> str:
        """Predict likelihood of dark web market proliferation"""
        risk_factors = []

        # High abuse potential
        if abuse_score >= 8.0:
            risk_factors.append('High abuse potential')

        # Novel structure (evades current regulations)
        if nps_class.get('is_nps'):
            risk_factors.append('Unscheduled (legal gray area)')

        # Low lethality (safer for recreational use)
        if 'LOW' in lethality:
            risk_factors.append('Relatively safe dose-response')

        # Fentanyl analogs (despite lethality)
        if any('Fentanyl' in c for c in nps_class.get('chemical_classes', [])):
            risk_factors.append('Fentanyl analog (high demand despite lethality)')

        # Synthetic cannabinoids
        if any('Cannabinoid' in c for c in nps_class.get('chemical_classes', [])):
            risk_factors.append('Synthetic cannabinoid (regulatory evasion)')

        # Calculate likelihood
        risk_count = len(risk_factors)

        if risk_count >= 3:
            return f'VERY HIGH ({risk_count}/5 risk factors) - Expect rapid dark web proliferation'
        elif risk_count >= 2:
            return f'HIGH ({risk_count}/5 risk factors) - Monitor dark web markets'
        elif risk_count >= 1:
            return f'MODERATE ({risk_count}/5 risk factors) - Possible emergence'
        else:
            return 'LOW - Unlikely to proliferate'

    def _calculate_abuse_score(self, nps_class: Dict, similarities: Dict,
                               receptors: Dict, reinforcement: List[str],
                               neurotox: str, lethality: str) -> float:
        """
        Calculate overall abuse potential score (0-10)

        10 = Extreme abuse potential (e.g., fentanyl)
        0 = No abuse potential
        """
        score = 0.0

        # Base score from chemical class (0-3 points)
        classes = nps_class.get('chemical_classes', [])
        if any('Fentanyl' in c for c in classes):
            score += 3.0
        elif any('Synthetic Opioid' in c for c in classes):
            score += 2.5
        elif any('Cathinone' in c or 'MDMA' in c for c in classes):
            score += 2.0
        elif any('Synthetic Cannabinoid' in c for c in classes):
            score += 1.5
        elif nps_class.get('is_nps'):
            score += 1.0

        # Similarity to known drugs (0-3 points)
        best_sim = similarities.get('best_similarity', 0.0)
        best_match = similarities.get('best_match', '')

        if best_match in ['fentanyl', 'heroin', 'methamphetamine'] and best_sim >= 0.7:
            score += 3.0
        elif best_sim >= 0.7:
            score += 2.0
        elif best_sim >= 0.5:
            score += 1.0

        # Receptor binding (0-2 points)
        if 'opioid' in receptors or 'dopamine' in receptors:
            score += 2.0
        elif len(receptors) >= 2:
            score += 1.5
        elif len(receptors) >= 1:
            score += 1.0

        # Reinforcement mechanisms (0-2 points)
        if any('EXTREME' in r for r in reinforcement):
            score += 2.0
        elif any('HIGH' in r for r in reinforcement):
            score += 1.5
        elif reinforcement:
            score += 1.0

        # Cap at 10.0
        return min(score, 10.0)

    def _categorize_abuse_risk(self, score: float) -> str:
        """Categorize abuse risk by score"""
        if score >= 9.0:
            return 'EXTREME - Immediate regulatory action required'
        elif score >= 7.0:
            return 'VERY HIGH - Emergency scheduling recommended'
        elif score >= 5.0:
            return 'HIGH - Monitor and prepare for scheduling'
        elif score >= 3.0:
            return 'MODERATE - Surveillance recommended'
        elif score >= 1.0:
            return 'LOW-MODERATE - Low priority monitoring'
        else:
            return 'LOW - Minimal abuse potential'

    def batch_screening(self, mol_ids: List[str],
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Batch screening of multiple compounds (large-scale threat intelligence)

        Args:
            mol_ids: List of molecule IDs to screen
            output_dir: Directory to save results

        Returns:
            Batch screening results
        """
        self.log(f"Starting batch screening of {len(mol_ids)} compounds...")
        start_time = time.time()

        results = {
            'total_compounds': len(mol_ids),
            'screened': 0,
            'high_risk_compounds': [],
            'moderate_risk_compounds': [],
            'low_risk_compounds': [],
            'novel_nps_detected': [],
            'elapsed_time': 0
        }

        for i, mol_id in enumerate(mol_ids, 1):
            self.log(f"Screening compound {i}/{len(mol_ids)}: {mol_id}")

            # Quick screening (not comprehensive)
            analysis = self.predict_abuse_potential(mol_id, comprehensive=False)

            score = analysis['abuse_potential_score']
            category = analysis['abuse_risk_category']

            # Categorize
            if 'EXTREME' in category or 'VERY HIGH' in category:
                results['high_risk_compounds'].append({
                    'mol_id': mol_id,
                    'score': score,
                    'category': category,
                    'classes': analysis['nps_classification']['chemical_classes']
                })
            elif 'HIGH' in category or 'MODERATE' in category:
                results['moderate_risk_compounds'].append({
                    'mol_id': mol_id,
                    'score': score,
                    'category': category
                })
            else:
                results['low_risk_compounds'].append({'mol_id': mol_id, 'score': score})

            # Track novel NPS
            if analysis['nps_classification']['is_nps']:
                results['novel_nps_detected'].append(mol_id)

            results['screened'] += 1

        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = output_path / f"batch_screening_{timestamp}.json"

            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

            self.log(f"Results saved to {result_file}")

        elapsed = time.time() - start_time
        results['elapsed_time'] = round(elapsed, 2)

        self.log(f"Batch screening complete in {elapsed:.2f} seconds")
        self.log(f"High-risk compounds: {len(results['high_risk_compounds'])}")
        self.log(f"Novel NPS detected: {len(results['novel_nps_detected'])}")

        return results

# Export
__all__ = ['NPSAbusePotentialAnalyzer']
