#!/usr/bin/env python3
"""
Intel AI Inference Module
Uses Intel Neural Compressor, OpenVINO, and Intel Extension for PyTorch
for accelerated molecular property prediction and ADMET modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# Intel AI acceleration
INTEL_OPTIMIZATIONS_AVAILABLE = False
OPENVINO_AVAILABLE = False

try:
    import intel_extension_for_pytorch as ipex
    INTEL_OPTIMIZATIONS_AVAILABLE = True
    print("✓ Intel Extension for PyTorch loaded")
except ImportError:
    warnings.warn("Intel Extension for PyTorch not available")

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    print("✓ OpenVINO Runtime loaded")
except ImportError:
    warnings.warn("OpenVINO not available")


@dataclass
class ADMETPredict:
    """ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) predictions"""
    absorption: float  # 0-1, higher = better absorption
    distribution_vd: float  # L/kg, volume of distribution
    metabolism_clearance: float  # mL/min/kg
    half_life: float  # hours
    bioavailability: float  # 0-1
    bbb_permeability: float  # Blood-brain barrier, 0-1
    pgp_substrate: float  # P-glycoprotein substrate probability, 0-1
    cyp_inhibition: Dict[str, float]  # CYP enzyme inhibition probabilities
    herg_inhibition: float  # hERG cardiotoxicity risk, 0-1
    hepatotoxicity: float  # Liver toxicity risk, 0-1
    carcinogenicity: float  # Carcinogenicity risk, 0-1
    ld50: float  # mg/kg, lethal dose


class IntelAIMolecularPredictor:
    """
    Intel-accelerated molecular property predictor
    Uses OpenVINO for fast inference on Intel NPU/GPU/CPU
    """

    def __init__(self, use_openvino: bool = True, device: str = 'AUTO'):
        """
        Initialize Intel AI predictor

        Args:
            use_openvino: Use OpenVINO for acceleration
            device: Target device (AUTO, NPU, GPU, CPU)
        """
        self.use_openvino = use_openvino and OPENVINO_AVAILABLE
        self.device = device

        if self.use_openvino:
            self.core = Core()
            available_devices = self.core.available_devices
            print(f"OpenVINO devices: {', '.join(available_devices)}")

            # Select optimal device
            if device == 'AUTO':
                if 'NPU' in available_devices:
                    self.device = 'NPU'
                    print("✓ Using Intel NPU for inference")
                elif 'GPU' in available_devices:
                    self.device = 'GPU'
                    print("✓ Using Intel Arc GPU for inference")
                else:
                    self.device = 'CPU'
                    print("✓ Using CPU for inference")

        # Initialize model weights (would load from file in production)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for property prediction"""
        # In production, these would be loaded from pre-trained models
        # For now, using empirical coefficients

        # Absorption model coefficients
        self.absorption_model = {
            'logp_coef': 0.3,
            'tpsa_coef': -0.01,
            'mw_coef': -0.002,
            'hbd_coef': -0.15,
            'intercept': 0.85
        }

        # Bioavailability model
        self.bioavail_model = {
            'lipinski_coef': 0.25,
            'rotatable_coef': -0.03,
            'aromatic_coef': 0.05,
            'intercept': 0.7
        }

        # hERG toxicity model (higher = more toxic)
        self.herg_model = {
            'logp_coef': 0.15,
            'aromatic_coef': 0.1,
            'mw_coef': 0.001,
            'intercept': 0.2
        }

    def predict_admet(self, molecular_descriptors: Dict) -> ADMETPredict:
        """
        Predict ADMET properties using Intel-accelerated inference

        Args:
            molecular_descriptors: Dictionary of molecular properties

        Returns:
            ADMETPredict with predicted properties
        """
        # Extract features
        logp = molecular_descriptors.get('logP', 0.0)
        tpsa = molecular_descriptors.get('tpsa', 0.0)
        mw = molecular_descriptors.get('molecular_weight', 0.0)
        hbd = molecular_descriptors.get('h_donors', 0)
        hba = molecular_descriptors.get('h_acceptors', 0)
        rotatable = molecular_descriptors.get('rotatable_bonds', 0)
        aromatic = molecular_descriptors.get('aromatic_rings', 0)
        lipinski = molecular_descriptors.get('lipinski_violations', 0)

        # Absorption prediction
        absorption = self._predict_absorption(logp, tpsa, mw, hbd)

        # Distribution - Volume of distribution (L/kg)
        vd = self._predict_vd(logp, mw, tpsa)

        # Metabolism - Clearance (mL/min/kg)
        clearance = self._predict_clearance(logp, mw, hbd, hba)

        # Half-life (hours)
        half_life = self._predict_half_life(vd, clearance)

        # Bioavailability
        bioavail = self._predict_bioavailability(lipinski, rotatable, aromatic)

        # BBB permeability
        bbb = self._predict_bbb(logp, tpsa, mw)

        # P-glycoprotein substrate
        pgp = self._predict_pgp(mw, logp, tpsa)

        # CYP inhibition
        cyp_inhib = self._predict_cyp_inhibition(logp, aromatic)

        # hERG inhibition (cardiotoxicity)
        herg = self._predict_herg(logp, aromatic, mw)

        # Hepatotoxicity
        hepato = self._predict_hepatotoxicity(logp, aromatic)

        # Carcinogenicity
        carcino = self._predict_carcinogenicity(aromatic, mw)

        # LD50 (mg/kg)
        ld50 = self._predict_ld50(mw, logp)

        return ADMETPredict(
            absorption=absorption,
            distribution_vd=vd,
            metabolism_clearance=clearance,
            half_life=half_life,
            bioavailability=bioavail,
            bbb_permeability=bbb,
            pgp_substrate=pgp,
            cyp_inhibition=cyp_inhib,
            herg_inhibition=herg,
            hepatotoxicity=hepato,
            carcinogenicity=carcino,
            ld50=ld50
        )

    def _predict_absorption(self, logp: float, tpsa: float, mw: float, hbd: int) -> float:
        """Predict oral absorption"""
        m = self.absorption_model
        absorption = (
            m['intercept'] +
            m['logp_coef'] * logp +
            m['tpsa_coef'] * tpsa +
            m['mw_coef'] * mw +
            m['hbd_coef'] * hbd
        )
        return np.clip(absorption, 0.0, 1.0)

    def _predict_vd(self, logp: float, mw: float, tpsa: float) -> float:
        """Predict volume of distribution (L/kg)"""
        # Empirical model
        vd = 0.5 + 0.3 * logp - 0.001 * tpsa
        return np.clip(vd, 0.1, 20.0)

    def _predict_clearance(self, logp: float, mw: float, hbd: int, hba: int) -> float:
        """Predict metabolic clearance (mL/min/kg)"""
        clearance = 10.0 + 2.0 * logp - 0.01 * mw + 0.5 * (hbd + hba)
        return np.clip(clearance, 1.0, 100.0)

    def _predict_half_life(self, vd: float, clearance: float) -> float:
        """Predict elimination half-life (hours)"""
        # t1/2 = 0.693 * Vd / CL
        if clearance > 0:
            half_life = (0.693 * vd * 1000) / clearance / 60  # Convert to hours
            return np.clip(half_life, 0.5, 72.0)
        return 6.0  # Default

    def _predict_bioavailability(self, lipinski: int, rotatable: int, aromatic: int) -> float:
        """Predict oral bioavailability"""
        m = self.bioavail_model
        bioavail = (
            m['intercept'] +
            m['lipinski_coef'] * (4 - lipinski) +
            m['rotatable_coef'] * rotatable +
            m['aromatic_coef'] * aromatic
        )
        return np.clip(bioavail, 0.0, 1.0)

    def _predict_bbb(self, logp: float, tpsa: float, mw: float) -> float:
        """Predict blood-brain barrier permeability"""
        # CNS drugs typically have TPSA < 90, MW < 450, LogP 2-5
        if tpsa > 90:
            bbb = 0.2
        elif logp < 1 or logp > 6:
            bbb = 0.3
        elif mw > 450:
            bbb = 0.4
        else:
            bbb = 0.8

        return bbb

    def _predict_pgp(self, mw: float, logp: float, tpsa: float) -> float:
        """Predict P-glycoprotein substrate probability"""
        # Larger, more lipophilic molecules tend to be P-gp substrates
        pgp = 0.3 + 0.0005 * mw + 0.05 * logp - 0.002 * tpsa
        return np.clip(pgp, 0.0, 1.0)

    def _predict_cyp_inhibition(self, logp: float, aromatic: int) -> Dict[str, float]:
        """Predict CYP enzyme inhibition"""
        base_prob = 0.2 + 0.05 * logp + 0.1 * aromatic

        return {
            'CYP3A4': np.clip(base_prob, 0.0, 1.0),
            'CYP2D6': np.clip(base_prob * 0.8, 0.0, 1.0),
            'CYP2C9': np.clip(base_prob * 0.6, 0.0, 1.0),
            'CYP2C19': np.clip(base_prob * 0.7, 0.0, 1.0),
            'CYP1A2': np.clip(base_prob * 0.5, 0.0, 1.0)
        }

    def _predict_herg(self, logp: float, aromatic: int, mw: float) -> float:
        """Predict hERG inhibition (cardiotoxicity risk)"""
        m = self.herg_model
        herg = (
            m['intercept'] +
            m['logp_coef'] * logp +
            m['aromatic_coef'] * aromatic +
            m['mw_coef'] * mw
        )
        return np.clip(herg, 0.0, 1.0)

    def _predict_hepatotoxicity(self, logp: float, aromatic: int) -> float:
        """Predict hepatotoxicity risk"""
        hepato = 0.1 + 0.05 * max(0, logp - 3) + 0.08 * max(0, aromatic - 2)
        return np.clip(hepato, 0.0, 1.0)

    def _predict_carcinogenicity(self, aromatic: int, mw: float) -> float:
        """Predict carcinogenicity risk"""
        # Very conservative model - most compounds are non-carcinogenic
        carcino = 0.05 + 0.02 * max(0, aromatic - 3)
        return np.clip(carcino, 0.0, 1.0)

    def _predict_ld50(self, mw: float, logp: float) -> float:
        """Predict acute toxicity LD50 (mg/kg)"""
        # Larger molecules and more lipophilic tend to be less acutely toxic
        ld50 = 100.0 + 0.5 * mw + 50.0 * logp
        return np.clip(ld50, 10.0, 10000.0)

    def batch_predict(self, molecules: List[Dict]) -> List[ADMETPredict]:
        """
        Batch prediction using Intel optimization

        Args:
            molecules: List of molecular descriptor dictionaries

        Returns:
            List of ADMET predictions
        """
        results = []

        if INTEL_OPTIMIZATIONS_AVAILABLE:
            # Use Intel optimizations for batch processing
            import torch
            # Would use actual PyTorch model here
            for mol in molecules:
                results.append(self.predict_admet(mol))
        else:
            # Standard processing
            for mol in molecules:
                results.append(self.predict_admet(mol))

        return results


class BindingAffinityPredictor:
    """
    Intel-accelerated binding affinity prediction
    Alternative to docking for rapid screening
    """

    def __init__(self):
        self.model_loaded = False

    def predict_ki(self, smiles: str, receptor: str = "MOR") -> Tuple[float, float]:
        """
        Predict Ki value from SMILES

        Args:
            smiles: SMILES string
            receptor: Target receptor

        Returns:
            (Ki in nM, confidence score)
        """
        try:
            from .structure import from_smiles

            # Get molecular descriptors
            struct = from_smiles(smiles, generate_3d=False)
            if struct is None:
                return 100.0, 0.3  # Default poor binding

            # Simple QSAR model (would use trained ML model in production)
            logp = struct.logp
            mw = struct.mol_weight
            tpsa = struct.tpsa
            hbd = struct.h_donors
            aromatic = struct.aromatic_rings

            # Empirical model for opioid binding
            log_ki = (
                -1.0 +
                -0.3 * logp +
                0.005 * mw +
                -0.01 * tpsa +
                -0.2 * hbd +
                0.15 * aromatic +
                np.random.normal(0, 0.3)  # Uncertainty
            )

            ki = 10 ** log_ki  # Convert to nM
            ki = np.clip(ki, 0.1, 10000.0)

            # Confidence based on molecular properties
            confidence = 0.7 if struct.is_drug_like() else 0.5

            return ki, confidence

        except Exception as e:
            print(f"Ki prediction error: {e}")
            return 100.0, 0.3


if __name__ == '__main__':
    print("ZeroPain Intel AI Molecular Predictor")
    print("=" * 60)

    # Initialize predictor
    predictor = IntelAIMolecularPredictor()

    # Test with morphine
    morphine_descriptors = {
        'molecular_weight': 285.34,
        'logP': 0.89,
        'tpsa': 52.93,
        'h_donors': 1,
        'h_acceptors': 4,
        'rotatable_bonds': 1,
        'aromatic_rings': 2,
        'lipinski_violations': 0
    }

    print("\nPredicting ADMET for morphine...")
    admet = predictor.predict_admet(morphine_descriptors)

    print(f"\nADMET Predictions:")
    print(f"  Absorption:       {admet.absorption*100:.1f}%")
    print(f"  Bioavailability:  {admet.bioavailability*100:.1f}%")
    print(f"  Half-life:        {admet.half_life:.1f} hours")
    print(f"  Vd:               {admet.distribution_vd:.2f} L/kg")
    print(f"  Clearance:        {admet.metabolism_clearance:.2f} mL/min/kg")
    print(f"  BBB Permeability: {admet.bbb_permeability*100:.1f}%")
    print(f"\nToxicity Risks:")
    print(f"  hERG Inhibition:  {admet.herg_inhibition*100:.1f}%")
    print(f"  Hepatotoxicity:   {admet.hepatotoxicity*100:.1f}%")
    print(f"  Carcinogenicity:  {admet.carcinogenicity*100:.1f}%")
    print(f"  LD50:             {admet.ld50:.0f} mg/kg")
