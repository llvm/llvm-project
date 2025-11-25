#!/usr/bin/env python3
"""
NMDA Agonist Analyzer - Antidepressant Properties Evaluation
Based on latest neuropharmacological research

Capabilities:
- NMDA receptor agonist/antagonist analysis
- Antidepressant property prediction
- Blood-Brain Barrier (BBB) permeability
- Neurotoxicity assessment
- Structure-Activity Relationship (SAR) analysis
- Literature-based efficacy evaluation

References:
- Autry et al. (2011) - NMDA antagonists and rapid antidepressant effects
- Zarate et al. (2006) - Ketamine for treatment-resistant depression
- Abdallah et al. (2015) - NMDA receptor modulation for depression
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sub_agents.rdkit_wrapper import RDKitAgent

class NMDAAgonistAnalyzer:
    def __init__(self):
        """Initialize NMDA agonist analyzer"""
        self.rdkit = RDKitAgent()

        # Known NMDA-related substructures (SMARTS patterns)
        self.nmda_patterns = {
            # Phenylcyclohexylamine scaffold (PCP-like)
            'pcp_scaffold': 'C1CCCCC1C(c2ccccc2)N',

            # Diarylethylamine (ketamine-like)
            'ketamine_scaffold': 'CC(=O)C(c1ccccc1Cl)N(C)C',

            # Glycine site binding
            'glycine_binding': 'NCC(=O)O',

            # Phencyclidine core
            'phencyclidine': 'C1CCC(CC1)(c2ccccc2)N3CCCCC3',

            # Memantine scaffold (adamantane)
            'memantine': 'C12CC3CC(C1)CC(C2)C3N',

            # D-serine (natural NMDA co-agonist)
            'd_serine': 'C(C(C(=O)O)N)O'
        }

        # NMDA receptor subtypes and binding sites
        self.receptor_sites = [
            'GluN1_glycine_binding',
            'GluN2A_glutamate_binding',
            'GluN2B_glutamate_binding',
            'GluN2C_glutamate_binding',
            'GluN2D_glutamate_binding',
            'PCP_binding_site'
        ]

        # Known NMDA modulators with antidepressant effects
        self.known_nmda_antidepressants = {
            'ketamine': {
                'smiles': 'CNC(=O)C(c1ccccc1Cl)N(C)C',
                'mechanism': 'NMDA antagonist',
                'efficacy': 'High (rapid-acting)',
                'onset': '2-4 hours',
                'duration': '1-2 weeks',
                'side_effects': 'Dissociation, abuse potential'
            },
            'esketamine': {
                'smiles': 'C[C@H](N)C(=O)c1ccccc1Cl',
                'mechanism': 'NMDA antagonist (S-enantiomer)',
                'efficacy': 'High (FDA approved)',
                'onset': '2-4 hours',
                'duration': '1-2 weeks',
                'side_effects': 'Dissociation, sedation'
            },
            'memantine': {
                'smiles': 'CC12CC3CC(C1)CC(N)(C3)C2',
                'mechanism': 'NMDA antagonist (uncompetitive)',
                'efficacy': 'Moderate',
                'onset': '1-2 weeks',
                'duration': 'Continuous',
                'side_effects': 'Dizziness, headache'
            },
            'd_cycloserine': {
                'smiles': 'C1=NOC(C1N)C(=O)O',
                'mechanism': 'Partial NMDA agonist',
                'efficacy': 'Moderate (augmentation)',
                'onset': 'Varies',
                'duration': 'Varies',
                'side_effects': 'Generally well-tolerated'
            }
        }

    def analyze_nmda_activity(self, mol_id: str) -> Dict[str, Any]:
        """
        Analyze molecule for NMDA receptor activity

        Args:
            mol_id: Molecule ID from RDKit agent

        Returns:
            Analysis of NMDA receptor interactions
        """
        results = {
            'molecule_id': mol_id,
            'nmda_activity_predicted': False,
            'scaffold_matches': [],
            'binding_site_predictions': {},
            'antidepressant_potential': 'Unknown',
            'recommendations': []
        }

        # Check substructure matches
        for name, pattern in self.nmda_patterns.items():
            match_result = self.rdkit.substructure_search(
                pattern=pattern,
                pattern_format='smarts'
            )

            if match_result.get('success'):
                matches = [m for m in match_result['matches'] if m['molecule_id'] == mol_id]
                if matches:
                    results['scaffold_matches'].append({
                        'scaffold': name,
                        'description': self._get_scaffold_description(name)
                    })
                    results['nmda_activity_predicted'] = True

        # Get molecular descriptors for prediction
        desc_result = self.rdkit.calculate_descriptors(
            mol_id=mol_id,
            descriptor_set='basic'
        )

        if desc_result.get('success'):
            descriptors = desc_result['descriptors']

            # Predict Blood-Brain Barrier penetration
            bbb_prediction = self._predict_bbb_penetration(descriptors)
            results['bbb_penetration'] = bbb_prediction

            # Predict binding affinity (simplified)
            binding_predictions = self._predict_binding_affinity(descriptors)
            results['binding_site_predictions'] = binding_predictions

            # Assess antidepressant potential
            results['antidepressant_potential'] = self._assess_antidepressant_potential(
                results['scaffold_matches'],
                bbb_prediction,
                binding_predictions,
                descriptors
            )

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results, descriptors)

        return results

    def _get_scaffold_description(self, scaffold_name: str) -> str:
        """Get description of scaffold significance"""
        descriptions = {
            'pcp_scaffold': 'Phencyclidine-like structure - strong NMDA antagonist activity',
            'ketamine_scaffold': 'Ketamine-like structure - rapid antidepressant effects',
            'glycine_binding': 'Glycine binding site modulat or',
            'phencyclidine': 'Classic PCP structure - dissociative anesthetic',
            'memantine': 'Memantine-like - uncompetitive NMDA antagonist',
            'd_serine': 'D-serine-like - natural NMDA co-agonist'
        }
        return descriptions.get(scaffold_name, 'NMDA-related structure')

    def _predict_bbb_penetration(self, descriptors: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict Blood-Brain Barrier penetration

        Uses simplified criteria:
        - MW < 400
        - LogP 1-5
        - TPSA < 90
        - H-bond donors < 3
        - H-bond acceptors < 8
        """
        mw = descriptors.get('MolecularWeight', 0)
        logp = descriptors.get('LogP', 0)
        tpsa = descriptors.get('TPSA', 0)
        hbd = descriptors.get('NumHDonors', 0)
        hba = descriptors.get('NumHAcceptors', 0)

        criteria_met = 0
        total_criteria = 5

        checks = {
            'mw_check': mw < 400,
            'logp_check': 1 < logp < 5,
            'tpsa_check': tpsa < 90,
            'hbd_check': hbd < 3,
            'hba_check': hba < 8
        }

        criteria_met = sum(checks.values())

        # Calculate penetration probability
        penetration_score = criteria_met / total_criteria

        if penetration_score >= 0.8:
            prediction = 'High'
        elif penetration_score >= 0.6:
            prediction = 'Moderate'
        else:
            prediction = 'Low'

        return {
            'prediction': prediction,
            'score': round(penetration_score, 2),
            'criteria': checks,
            'confidence': 'Medium (simplified model)'
        }

    def _predict_binding_affinity(self, descriptors: Dict[str, float]) -> Dict[str, str]:
        """
        Predict NMDA receptor binding (simplified)

        In production, this would use QSAR models or molecular docking
        """
        mw = descriptors.get('MolecularWeight', 0)
        logp = descriptors.get('LogP', 0)
        aromatic_rings = descriptors.get('NumAromaticRings', 0)

        predictions = {}

        # Simplified heuristics (replace with ML model in production)
        if 200 < mw < 350 and 2 < logp < 5 and aromatic_rings >= 1:
            predictions['PCP_binding_site'] = 'Moderate affinity predicted'
        elif mw < 200:
            predictions['GluN1_glycine_binding'] = 'Possible glycine site interaction'
        else:
            predictions['general'] = 'Insufficient data for binding prediction'

        return predictions

    def _assess_antidepressant_potential(self, scaffolds: List[Dict],
                                        bbb: Dict, binding: Dict,
                                        descriptors: Dict) -> str:
        """Assess antidepressant potential based on analysis"""

        # High potential indicators
        high_indicators = [
            any('ketamine' in s['scaffold'] for s in scaffolds),
            any('pcp' in s['scaffold'] for s in scaffolds),
            bbb['prediction'] == 'High',
            bool(binding.get('PCP_binding_site'))
        ]

        # Moderate potential indicators
        moderate_indicators = [
            any('memantine' in s['scaffold'] for s in scaffolds),
            bbb['prediction'] == 'Moderate',
            descriptors.get('LogP', 0) > 2
        ]

        if sum(high_indicators) >= 2:
            return 'High (ketamine-like profile)'
        elif sum(high_indicators) >= 1 or sum(moderate_indicators) >= 2:
            return 'Moderate (NMDA modulation likely)'
        else:
            return 'Low (insufficient NMDA activity indicators)'

    def _generate_recommendations(self, analysis: Dict, descriptors: Dict) -> List[str]:
        """Generate recommendations for further analysis"""
        recommendations = []

        # BBB recommendations
        if analysis['bbb_penetration']['prediction'] == 'Low':
            recommendations.append(
                'Consider structural modifications to improve BBB penetration (reduce TPSA, optimize LogP)'
            )

        # NMDA activity recommendations
        if not analysis['scaffold_matches']:
            recommendations.append(
                'No known NMDA-related scaffolds detected. Consider:' +
                '(1) Molecular docking studies, (2) QSAR model evaluation, (3) Literature search for novel mechanisms'
            )

        # Efficacy recommendations
        if analysis['antidepressant_potential'].startswith('High'):
            recommendations.append(
                'High antidepressant potential indicated. Next steps: ' +
                '(1) In vitro NMDA receptor binding assays, (2) Electrophysiology studies, (3) Animal behavior models'
            )
            recommendations.append(
                'Safety assessment critical: Evaluate abuse potential, cognitive effects, and neurotoxicity'
            )

        # General recommendations
        recommendations.append(
            'Consult latest literature: PubMed search for "NMDA + antidepressant + [compound class]"'
        )

        recommendations.append(
            'Consider enantiomeric effects (S-ketamine > R-ketamine for rapid antidepressant action)'
        )

        return recommendations

    def compare_with_known_antidepressants(self, mol_id: str) -> Dict[str, Any]:
        """
        Compare molecule with known NMDA-based antidepressants

        Args:
            mol_id: Molecule to analyze

        Returns:
            Comparison results with known compounds
        """
        results = {
            'molecule_id': mol_id,
            'comparisons': []
        }

        # Load all known antidepressants into RDKit
        reference_mols = {}
        for name, data in self.known_nmda_antidepressants.items():
            parse_result = self.rdkit.parse_molecule(
                structure=data['smiles'],
                format='smiles',
                name=f"Reference_{name}"
            )
            if parse_result.get('success'):
                reference_mols[name] = parse_result['molecule_id']

        # Generate fingerprint for query molecule
        fp_result = self.rdkit.generate_fingerprint(
            mol_id=mol_id,
            fp_type='morgan',
            radius=2,
            n_bits=2048
        )

        if not fp_result.get('success'):
            return {
                'success': False,
                'error': 'Failed to generate fingerprint for query molecule'
            }

        # Compare with each reference
        for name, ref_mol_id in reference_mols.items():
            # Generate reference fingerprint
            ref_fp = self.rdkit.generate_fingerprint(
                mol_id=ref_mol_id,
                fp_type='morgan',
                radius=2,
                n_bits=2048
            )

            # Calculate similarity
            similarity_result = self.rdkit.similarity_search(
                query_mol_id=mol_id,
                target_mol_ids=[ref_mol_id],
                fp_type='morgan',
                metric='tanimoto'
            )

            if similarity_result.get('success') and similarity_result['results']:
                similarity_score = similarity_result['results'][0]['similarity']

                comparison = {
                    'reference': name,
                    'mechanism': self.known_nmda_antidepressants[name]['mechanism'],
                    'efficacy': self.known_nmda_antidepressants[name]['efficacy'],
                    'similarity_score': similarity_score,
                    'interpretation': self._interpret_similarity(similarity_score, name)
                }

                results['comparisons'].append(comparison)

        # Sort by similarity
        results['comparisons'].sort(key=lambda x: x['similarity_score'], reverse=True)

        # Add overall assessment
        if results['comparisons']:
            best_match = results['comparisons'][0]
            results['best_match'] = best_match['reference']
            results['best_similarity'] = best_match['similarity_score']

            if best_match['similarity_score'] >= 0.7:
                results['assessment'] = f"High similarity to {best_match['reference']} - likely similar mechanism"
            elif best_match['similarity_score'] >= 0.5:
                results['assessment'] = f"Moderate similarity to {best_match['reference']} - may share some properties"
            else:
                results['assessment'] = "Novel structure - distinct from known NMDA antidepressants"

        return results

    def _interpret_similarity(self, score: float, reference: str) -> str:
        """Interpret similarity score"""
        if score >= 0.85:
            return f"Very high similarity - likely analog of {reference}"
        elif score >= 0.7:
            return f"High similarity - may share mechanism with {reference}"
        elif score >= 0.5:
            return f"Moderate similarity - some structural features of {reference}"
        else:
            return f"Low similarity - structurally distinct from {reference}"

    def comprehensive_analysis(self, mol_id: str) -> Dict[str, Any]:
        """
        Comprehensive NMDA agonist/antagonist antidepressant analysis

        Args:
            mol_id: Molecule ID

        Returns:
            Complete analysis report
        """
        # NMDA activity analysis
        nmda_analysis = self.analyze_nmda_activity(mol_id)

        # Comparison with known compounds
        comparison = self.compare_with_known_antidepressants(mol_id)

        # Drug-likeness
        drug_likeness = self.rdkit.drug_likeness_analysis(mol_id)

        # Compile comprehensive report
        report = {
            'molecule_id': mol_id,
            'nmda_analysis': nmda_analysis,
            'known_compound_comparison': comparison,
            'drug_likeness': drug_likeness,
            'overall_assessment': self._generate_overall_assessment(
                nmda_analysis, comparison, drug_likeness
            )
        }

        return report

    def _generate_overall_assessment(self, nmda: Dict, comparison: Dict,
                                    drug_likeness: Dict) -> Dict[str, Any]:
        """Generate overall assessment"""
        assessment = {
            'antidepressant_potential': nmda.get('antidepressant_potential', 'Unknown'),
            'nmda_activity': 'Predicted' if nmda.get('nmda_activity_predicted') else 'Unlikely',
            'development_priority': 'Unknown',
            'key_findings': [],
            'next_steps': []
        }

        # Key findings
        if nmda.get('scaffold_matches'):
            assessment['key_findings'].append(
                f"Contains {len(nmda['scaffold_matches'])} NMDA-related scaffold(s)"
            )

        if comparison.get('best_similarity', 0) >= 0.7:
            assessment['key_findings'].append(
                f"High similarity to {comparison['best_match']} ({comparison['best_similarity']:.2f})"
            )

        if drug_likeness.get('success'):
            overall_dl = drug_likeness.get('overall_drug_likeness', '')
            assessment['key_findings'].append(f"Drug-likeness: {overall_dl}")

        # Development priority
        high_priority_indicators = [
            nmda.get('antidepressant_potential', '').startswith('High'),
            comparison.get('best_similarity', 0) >= 0.7,
            drug_likeness.get('overall_drug_likeness') == 'Good',
            nmda.get('bbb_penetration', {}).get('prediction') == 'High'
        ]

        if sum(high_priority_indicators) >= 3:
            assessment['development_priority'] = 'HIGH - Proceed to in vitro studies'
        elif sum(high_priority_indicators) >= 2:
            assessment['development_priority'] = 'MODERATE - Consider optimization'
        else:
            assessment['development_priority'] = 'LOW - Requires structural modification'

        # Next steps
        if assessment['development_priority'].startswith('HIGH'):
            assessment['next_steps'] = [
                '1. NMDA receptor binding assays (GluN2B preferably)',
                '2. Patch-clamp electrophysiology',
                '3. BBB permeability assay (PAMPA or Caco-2)',
                '4. Cytotoxicity screening',
                '5. Animal behavioral models (forced swim test, learned helplessness)',
                '6. Literature review for similar structures',
                '7. Patent landscape analysis'
            ]
        else:
            assessment['next_steps'] = [
                '1. Structure optimization to improve NMDA binding',
                '2. BBB penetration enhancement',
                '3. Drug-likeness improvement',
                '4. Computational docking studies',
                '5. Literature search for optimization strategies'
            ]

        return assessment

# Export
__all__ = ['NMDAAgonistAnalyzer']
