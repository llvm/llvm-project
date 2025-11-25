#!/usr/bin/env python3
"""
RDKit Cheminformatics Agent
Molecular structure analysis, drug discovery, chemical fingerprinting

Capabilities:
- Molecular structure parsing (SMILES, SDF, MOL, InChI)
- 2D/3D conformer generation
- Molecular fingerprints for ML (Morgan, MACCS, RDK, etc.)
- Descriptor calculation (>200 molecular descriptors)
- Substructure/similarity searching
- Drug-likeness analysis (Lipinski, Veber, etc.)
- Structure-activity relationship analysis

Dependencies: rdkit, pandas, numpy
"""

import os
import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RDKitAgent:
    def __init__(self):
        """Initialize RDKit cheminformatics agent"""
        self.data_dir = Path.home() / ".dsmil" / "rdkit"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.molecules_dir = self.data_dir / "molecules"
        self.molecules_dir.mkdir(exist_ok=True)

        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Track loaded molecules
        self.molecules = {}
        self._load_molecule_registry()

        # Check if RDKit is available
        self.rdkit_available = self._check_rdkit()

    def _check_rdkit(self) -> bool:
        """Check if RDKit is installed"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """Check if agent is available"""
        return self.rdkit_available

    def _load_molecule_registry(self):
        """Load molecule registry from disk"""
        registry_file = self.data_dir / "molecules.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.molecules = json.load(f)

    def _save_molecule_registry(self):
        """Save molecule registry to disk"""
        registry_file = self.data_dir / "molecules.json"
        with open(registry_file, 'w') as f:
            json.dump(self.molecules, f, indent=2)

    def parse_molecule(self, structure: str,
                       format: str = "smiles",
                       name: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse molecular structure from various formats

        Args:
            structure: Molecular structure (SMILES string, InChI, etc.)
            format: 'smiles', 'inchi', 'mol', 'sdf'
            name: Optional name for molecule

        Returns:
            Dict with molecule info and ID
        """
        if not self.rdkit_available:
            return {
                "success": False,
                "error": "RDKit not installed. Install: pip install rdkit"
            }

        from rdkit import Chem

        try:
            # Parse based on format
            if format == "smiles":
                mol = Chem.MolFromSmiles(structure)
            elif format == "inchi":
                mol = Chem.MolFromInchi(structure)
            elif format in ["mol", "sdf"]:
                mol = Chem.MolFromMolBlock(structure)
            else:
                return {
                    "success": False,
                    "error": f"Unknown format: {format}"
                }

            if mol is None:
                return {
                    "success": False,
                    "error": f"Failed to parse molecule from {format}"
                }

            # Generate ID
            mol_id = f"mol_{len(self.molecules) + 1}"

            # Get basic properties
            from rdkit.Chem import Descriptors

            props = {
                "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": Descriptors.MolWt(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": Descriptors.RingCount(mol),
                "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "num_hbd": Descriptors.NumHDonors(mol),
                "num_hba": Descriptors.NumHAcceptors(mol),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol)
            }

            # Store molecule
            if not name:
                name = f"Molecule_{len(self.molecules) + 1}"

            # Save as SMILES
            canonical_smiles = Chem.MolToSmiles(mol)

            self.molecules[mol_id] = {
                "id": mol_id,
                "name": name,
                "smiles": canonical_smiles,
                "input_format": format,
                "properties": props
            }

            # Save MOL file
            mol_file = self.molecules_dir / f"{mol_id}.mol"
            with open(mol_file, 'w') as f:
                f.write(Chem.MolToMolBlock(mol))

            self._save_molecule_registry()

            return {
                "success": True,
                "molecule_id": mol_id,
                "name": name,
                "smiles": canonical_smiles,
                "properties": props
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse molecule: {str(e)}"
            }

    def calculate_descriptors(self, mol_id: str,
                             descriptor_set: str = "basic") -> Dict[str, Any]:
        """
        Calculate molecular descriptors

        Args:
            mol_id: Molecule ID
            descriptor_set: 'basic', 'all', 'lipinski', 'custom'

        Returns:
            Dict with calculated descriptors
        """
        if not self.rdkit_available:
            return {
                "success": False,
                "error": "RDKit not installed"
            }

        if mol_id not in self.molecules:
            return {
                "success": False,
                "error": f"Molecule {mol_id} not found"
            }

        from rdkit import Chem
        from rdkit.Chem import Descriptors

        # Load molecule
        mol_file = self.molecules_dir / f"{mol_id}.mol"
        mol = Chem.MolFromMolFile(str(mol_file))

        if mol is None:
            return {
                "success": False,
                "error": "Failed to load molecule"
            }

        descriptors = {}

        if descriptor_set == "basic":
            descriptors = {
                "MolecularWeight": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                "FractionCsp3": Descriptors.FractionCsp3(mol)
            }

        elif descriptor_set == "lipinski":
            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            lipinski_violations = sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10
            ])

            descriptors = {
                "MolecularWeight": mw,
                "LogP": logp,
                "HBondDonors": hbd,
                "HBondAcceptors": hba,
                "LipinskiViolations": lipinski_violations,
                "PassesLipinski": lipinski_violations <= 1
            }

        elif descriptor_set == "all":
            # Calculate all available descriptors
            for name, func in Descriptors.descList:
                try:
                    descriptors[name] = func(mol)
                except:
                    pass

        return {
            "success": True,
            "molecule_id": mol_id,
            "descriptor_set": descriptor_set,
            "descriptors": descriptors,
            "count": len(descriptors)
        }

    def generate_fingerprint(self, mol_id: str,
                            fp_type: str = "morgan",
                            radius: int = 2,
                            n_bits: int = 2048) -> Dict[str, Any]:
        """
        Generate molecular fingerprint for ML/similarity analysis

        Args:
            mol_id: Molecule ID
            fp_type: 'morgan', 'maccs', 'rdk', 'atompair', 'topological'
            radius: Radius for Morgan fingerprints (default: 2)
            n_bits: Number of bits (default: 2048)

        Returns:
            Dict with fingerprint info
        """
        if not self.rdkit_available:
            return {
                "success": False,
                "error": "RDKit not installed"
            }

        if mol_id not in self.molecules:
            return {
                "success": False,
                "error": f"Molecule {mol_id} not found"
            }

        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import DataStructs
        import numpy as np

        # Load molecule
        mol_file = self.molecules_dir / f"{mol_id}.mol"
        mol = Chem.MolFromMolFile(str(mol_file))

        if mol is None:
            return {
                "success": False,
                "error": "Failed to load molecule"
            }

        try:
            # Generate fingerprint based on type
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == "maccs":
                fp = MACCSkeys.GenMACCSKeys(mol)
                n_bits = 167  # MACCS is always 167 bits
            elif fp_type == "rdk":
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            elif fp_type == "atompair":
                fp = AllChem.GetAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            elif fp_type == "topological":
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            else:
                return {
                    "success": False,
                    "error": f"Unknown fingerprint type: {fp_type}"
                }

            # Convert to bit string
            fp_array = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, fp_array)

            # Save fingerprint
            fp_file = self.results_dir / f"{mol_id}_{fp_type}_fp.json"
            with open(fp_file, 'w') as f:
                json.dump({
                    "molecule_id": mol_id,
                    "fp_type": fp_type,
                    "radius": radius,
                    "n_bits": n_bits,
                    "fingerprint": fp_array.tolist()
                }, f)

            return {
                "success": True,
                "molecule_id": mol_id,
                "fp_type": fp_type,
                "n_bits": n_bits,
                "radius": radius if fp_type == "morgan" else None,
                "fingerprint_file": str(fp_file),
                "on_bits": int(fp_array.sum())
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate fingerprint: {str(e)}"
            }

    def similarity_search(self, query_mol_id: str,
                         target_mol_ids: Optional[List[str]] = None,
                         fp_type: str = "morgan",
                         metric: str = "tanimoto") -> Dict[str, Any]:
        """
        Calculate similarity between molecules

        Args:
            query_mol_id: Query molecule ID
            target_mol_ids: Target molecule IDs (None = all)
            fp_type: Fingerprint type
            metric: 'tanimoto', 'dice', 'cosine'

        Returns:
            Dict with similarity scores
        """
        if not self.rdkit_available:
            return {
                "success": False,
                "error": "RDKit not installed"
            }

        if query_mol_id not in self.molecules:
            return {
                "success": False,
                "error": f"Query molecule {query_mol_id} not found"
            }

        if target_mol_ids is None:
            target_mol_ids = [mid for mid in self.molecules.keys() if mid != query_mol_id]

        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import DataStructs

        # Load query molecule
        query_mol_file = self.molecules_dir / f"{query_mol_id}.mol"
        query_mol = Chem.MolFromMolFile(str(query_mol_file))

        if query_mol is None:
            return {
                "success": False,
                "error": "Failed to load query molecule"
            }

        # Generate query fingerprint
        if fp_type == "morgan":
            query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        elif fp_type == "maccs":
            query_fp = MACCSkeys.GenMACCSKeys(query_mol)
        else:
            query_fp = Chem.RDKFingerprint(query_mol)

        # Calculate similarities
        similarities = []

        for target_id in target_mol_ids:
            if target_id not in self.molecules:
                continue

            target_mol_file = self.molecules_dir / f"{target_id}.mol"
            target_mol = Chem.MolFromMolFile(str(target_mol_file))

            if target_mol is None:
                continue

            # Generate target fingerprint
            if fp_type == "morgan":
                target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
            elif fp_type == "maccs":
                target_fp = MACCSkeys.GenMACCSKeys(target_mol)
            else:
                target_fp = Chem.RDKFingerprint(target_mol)

            # Calculate similarity
            if metric == "tanimoto":
                similarity = DataStructs.TanimotoSimilarity(query_fp, target_fp)
            elif metric == "dice":
                similarity = DataStructs.DiceSimilarity(query_fp, target_fp)
            elif metric == "cosine":
                similarity = DataStructs.CosineSimilarity(query_fp, target_fp)
            else:
                similarity = DataStructs.TanimotoSimilarity(query_fp, target_fp)

            similarities.append({
                "molecule_id": target_id,
                "name": self.molecules[target_id]["name"],
                "similarity": round(similarity, 4)
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "success": True,
            "query_molecule": query_mol_id,
            "fp_type": fp_type,
            "metric": metric,
            "results": similarities,
            "count": len(similarities)
        }

    def drug_likeness_analysis(self, mol_id: str) -> Dict[str, Any]:
        """
        Analyze drug-likeness (Lipinski, Veber, etc.)

        Args:
            mol_id: Molecule ID

        Returns:
            Dict with drug-likeness analysis
        """
        if not self.rdkit_available:
            return {
                "success": False,
                "error": "RDKit not installed"
            }

        if mol_id not in self.molecules:
            return {
                "success": False,
                "error": f"Molecule {mol_id} not found"
            }

        from rdkit import Chem
        from rdkit.Chem import Descriptors

        # Load molecule
        mol_file = self.molecules_dir / f"{mol_id}.mol"
        mol = Chem.MolFromMolFile(str(mol_file))

        if mol is None:
            return {
                "success": False,
                "error": "Failed to load molecule"
            }

        # Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])

        # Veber's Rules
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)

        veber_violations = sum([
            rotatable_bonds > 10,
            tpsa > 140
        ])

        # QED (Quantitative Estimate of Drug-likeness)
        try:
            from rdkit.Chem import QED
            qed_score = QED.qed(mol)
        except:
            qed_score = None

        return {
            "success": True,
            "molecule_id": mol_id,
            "lipinski": {
                "molecular_weight": mw,
                "logp": logp,
                "hbd": hbd,
                "hba": hba,
                "violations": lipinski_violations,
                "passes": lipinski_violations <= 1
            },
            "veber": {
                "rotatable_bonds": rotatable_bonds,
                "tpsa": tpsa,
                "violations": veber_violations,
                "passes": veber_violations == 0
            },
            "qed": qed_score,
            "overall_drug_likeness": "Good" if lipinski_violations <= 1 and veber_violations == 0 else "Poor"
        }

    def substructure_search(self, pattern: str,
                           pattern_format: str = "smarts") -> Dict[str, Any]:
        """
        Search for substructure pattern in all molecules

        Args:
            pattern: Substructure pattern
            pattern_format: 'smarts' or 'smiles'

        Returns:
            Dict with matching molecules
        """
        if not self.rdkit_available:
            return {
                "success": False,
                "error": "RDKit not installed"
            }

        from rdkit import Chem

        # Parse pattern
        if pattern_format == "smarts":
            pattern_mol = Chem.MolFromSmarts(pattern)
        else:
            pattern_mol = Chem.MolFromSmiles(pattern)

        if pattern_mol is None:
            return {
                "success": False,
                "error": f"Failed to parse pattern as {pattern_format}"
            }

        # Search all molecules
        matches = []

        for mol_id, mol_info in self.molecules.items():
            mol_file = self.molecules_dir / f"{mol_id}.mol"
            mol = Chem.MolFromMolFile(str(mol_file))

            if mol is None:
                continue

            if mol.HasSubstructMatch(pattern_mol):
                match_atoms = mol.GetSubstructMatch(pattern_mol)
                matches.append({
                    "molecule_id": mol_id,
                    "name": mol_info["name"],
                    "smiles": mol_info["smiles"],
                    "match_atoms": list(match_atoms)
                })

        return {
            "success": True,
            "pattern": pattern,
            "pattern_format": pattern_format,
            "matches": matches,
            "count": len(matches)
        }

    def list_molecules(self) -> Dict[str, Any]:
        """List all loaded molecules"""
        return {
            "success": True,
            "molecules": list(self.molecules.values()),
            "count": len(self.molecules)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "available": self.is_available(),
            "rdkit_installed": self.rdkit_available,
            "molecules_loaded": len(self.molecules),
            "storage_path": str(self.data_dir)
        }

# Export
__all__ = ['RDKitAgent']
