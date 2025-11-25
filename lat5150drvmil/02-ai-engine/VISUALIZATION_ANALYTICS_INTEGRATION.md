# Visualization & Analytics Integration

Complete integration of advanced data visualization and analytics capabilities into the DSMIL AI Engine.

## Overview

This integration adds 6 powerful specialized agents for:
1. **Geospatial Analytics** - OSINT, threat-intel mapping, infrastructure visualization
2. **Cheminformatics** - Drug discovery, molecular analysis (RDKit)
3. **Data Visualization & ML** - Pattern recognition, interactive visualizations (PRT-style)
4. **GPU Virtualization** - KVM/Xen GPU passthrough management (MxGPU)
5. **NMDA Antidepressant Analysis** - NMDA receptor agonist/antagonist screening for novel antidepressants
6. **NPS Abuse Potential Analysis** - PROACTIVE novel psychoactive substance threat intelligence

All agents are **LOCAL-FIRST**, **DSMIL-attested**, and **zero-cost**.

## Architecture

### Components

Each integration follows the proven NotebookLM pattern:

```
02-ai-engine/
├── sub_agents/
│   ├── geospatial_wrapper.py          # OpenSphere-inspired geospatial analytics
│   ├── rdkit_wrapper.py               # RDKit cheminformatics
│   ├── prt_visualization_wrapper.py   # PRT-style data visualization & ML
│   ├── mxgpu_wrapper.py               # MxGPU GPU virtualization
│   ├── nmda_agonist_analyzer.py       # NMDA antidepressant analysis
│   └── nps_abuse_potential_analyzer.py # NPS abuse potential prediction
│
├── geospatial_cli.py                  # Geospatial CLI
├── rdkit_cli.py                       # RDKit CLI
├── prt_cli.py                         # PRT visualization CLI
├── mxgpu_cli.py                       # MxGPU CLI
├── nmda_cli.py                        # NMDA analyzer CLI
├── nps_cli.py                         # NPS analyzer CLI
│
├── smart_router.py                    # Auto-detection patterns (UPDATED)
├── unified_orchestrator.py            # Routing logic (UPDATED)
└── VISUALIZATION_ANALYTICS_INTEGRATION.md
```

### Storage Locations

All agent data is stored locally:

```
~/.dsmil/
├── geospatial/           # Geospatial datasets and maps
│   ├── datasets/         # GeoJSON datasets
│   ├── maps/             # Generated HTML maps
│   └── datasets.json     # Dataset registry
│
├── rdkit/                # RDKit molecules and results
│   ├── molecules/        # MOL files
│   ├── results/          # Fingerprints, descriptors
│   └── molecules.json    # Molecule registry
│
├── prt/                  # PRT datasets and visualizations
│   ├── datasets/         # CSV datasets
│   ├── visualizations/   # PNG/HTML visualizations
│   ├── models/           # Trained ML models
│   ├── datasets.json     # Dataset registry
│   └── models.json       # Model registry
│
├── mxgpu/                # MxGPU configurations
│   ├── configs/          # VM configurations
│   ├── logs/             # Operation logs
│   └── gpu_configs.json  # GPU registry
│
├── nmda_results/         # NMDA antidepressant analysis results
│   ├── analyses/         # Individual analysis JSON files
│   ├── batch_reports/    # Batch screening reports
│   └── top_candidates.json # Top-ranked compounds
│
└── nps_results/          # NPS abuse potential analysis results
    ├── threat_intel/     # High-risk compound reports
    ├── batch_screens/    # Large-scale screening results
    └── risk_database.json # Abuse potential scores
```

## 1. Geospatial Analytics (OpenSphere-Inspired)

### Capabilities

- **Data Loading**: KML, GeoJSON, Shapefiles, GPX, CSV (with lat/lon)
- **Map Creation**: Interactive 2D/3D maps (Folium, PyDeck, Plotly)
- **Threat Intelligence**: Hotspot analysis, density mapping
- **OSINT Mapping**: Visualize threat actors, infrastructure
- **Temporal Analytics**: Time-series geospatial analysis

### Dependencies

```bash
pip install geopandas folium pydeck plotly shapely pandas
```

### Natural Language Usage

The Smart Router automatically detects geospatial queries:

```python
from unified_orchestrator import UnifiedAIOrchestrator

orch = UnifiedAIOrchestrator()

# Load geospatial data
result = orch.query(
    "Load threat intel data",
    file_path="/path/to/threat_data.geojson",
    dataset_name="APT Campaign 2024"
)

# Create threat intelligence map
result = orch.query(
    "Map this threat intelligence",
    dataset_ids=["geo_1"],
    map_type="folium",
    title="APT Campaign Infrastructure"
)

# Analyze hotspots
result = orch.query(
    "Analyze threat intel hotspots",
    dataset_id="geo_1",
    analysis_type="hotspot"
)
```

### Detection Keywords

**Smart Router auto-routes on:**
- Actions: `map`, `visualize geo`, `plot location`, `show on map`
- Artifacts: `threat intel`, `osint`, `infrastructure`, `coordinates`
- Formats: `kml`, `geojson`, `shapefile`, `gpx`
- Modes: `geospatial`, `osint mapping`, `threat intel map`

### API Reference

```python
from sub_agents.geospatial_wrapper import GeospatialAgent

geo = GeospatialAgent()

# Load data
result = geo.load_data(
    file_path="/path/to/data.geojson",
    dataset_name="My Dataset"
)
# Returns: {"success": True, "dataset_id": "geo_1", ...}

# Create map
result = geo.create_map(
    dataset_ids=["geo_1"],
    map_type="folium",  # or "pydeck", "plotly"
    title="My Map",
    style="dark"  # or "default", "satellite", "light"
)
# Returns: {"success": True, "file_path": "~/.dsmil/geospatial/maps/map_1.html"}

# Threat intel analysis
result = geo.threat_intel_analysis(
    dataset_id="geo_1",
    analysis_type="hotspot"  # or "density"
)

# List datasets
result = geo.list_datasets()

# Get status
status = geo.get_status()
```

## 2. RDKit Cheminformatics

### Capabilities

- **Molecular Parsing**: SMILES, InChI, SDF, MOL formats
- **Descriptors**: 200+ molecular properties (MW, LogP, TPSA, etc.)
- **Fingerprints**: Morgan, MACCS, RDK, AtomPair, Topological
- **Similarity Search**: Tanimoto, Dice, Cosine metrics
- **Drug-Likeness**: Lipinski, Veber, QED analysis
- **Substructure Search**: SMARTS pattern matching

### Dependencies

```bash
pip install rdkit pandas numpy
```

### Natural Language Usage

```python
# Parse molecule
result = orch.query(
    "Analyze this molecule",
    structure="CCO",  # Ethanol SMILES
    format="smiles",
    name="Ethanol"
)

# Calculate descriptors
result = orch.query(
    "Calculate drug likeness",
    mol_id="mol_1"
)

# Generate fingerprint
result = orch.query(
    "Generate molecular fingerprint",
    mol_id="mol_1",
    fp_type="morgan",
    radius=2,
    n_bits=2048
)

# Similarity search
result = orch.query(
    "Find similar compounds",
    query_mol_id="mol_1",
    fp_type="morgan",
    metric="tanimoto"
)
```

### Detection Keywords

**Smart Router auto-routes on:**
- Actions: `analyze molecule`, `calculate descriptor`, `drug likeness`
- Artifacts: `molecule`, `compound`, `smiles`, `chemical`, `drug`
- Operations: `parse smiles`, `fingerprint`, `substructure`, `lipinski`
- Modes: `cheminformatics`, `rdkit`, `drug discovery`

### API Reference

```python
from sub_agents.rdkit_wrapper import RDKitAgent

rdkit = RDKitAgent()

# Parse molecule
result = rdkit.parse_molecule(
    structure="CCO",
    format="smiles",  # or "inchi", "mol", "sdf"
    name="Ethanol"
)

# Calculate descriptors
result = rdkit.calculate_descriptors(
    mol_id="mol_1",
    descriptor_set="basic"  # or "all", "lipinski"
)

# Generate fingerprint
result = rdkit.generate_fingerprint(
    mol_id="mol_1",
    fp_type="morgan",  # or "maccs", "rdk", "atompair"
    radius=2,
    n_bits=2048
)

# Similarity search
result = rdkit.similarity_search(
    query_mol_id="mol_1",
    target_mol_ids=None,  # None = search all
    fp_type="morgan",
    metric="tanimoto"
)

# Drug-likeness
result = rdkit.drug_likeness_analysis(mol_id="mol_1")

# Substructure search
result = rdkit.substructure_search(
    pattern="c1ccccc1",  # Benzene ring
    pattern_format="smarts"
)
```

## 3. PRT-Style Data Visualization & ML

### Capabilities

- **Data Loading**: CSV, Excel files
- **Exploratory Analysis**: Statistical summaries, distributions
- **Visualizations**: Correlation, distribution, scatter, pairplot, boxplot
- **Classification**: Random Forest, SVM, Logistic, KNN, Decision Tree
- **Clustering**: K-Means, DBSCAN, Hierarchical
- **Dimensionality Reduction**: PCA, t-SNE

### Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly joblib
```

### Natural Language Usage

```python
# Load dataset
result = orch.query(
    "Load dataset for analysis",
    file_path="/path/to/data.csv",
    name="Customer Data",
    target_column="purchase"  # For supervised learning
)

# Explore dataset
result = orch.query(
    "Explore data statistics",
    dataset_id="ds_1"
)

# Visualize
result = orch.query(
    "Visualize data correlations",
    dataset_id="ds_1",
    viz_type="correlation"
)

# Train classifier
result = orch.query(
    "Train a classifier on this data",
    dataset_id="ds_1",
    algorithm="random_forest",
    test_size=0.2
)

# Clustering
result = orch.query(
    "Cluster this data",
    dataset_id="ds_1",
    n_clusters=3,
    algorithm="kmeans"
)

# Dimensionality reduction
result = orch.query(
    "Reduce dimensions with PCA",
    dataset_id="ds_1",
    method="pca",
    n_components=2
)
```

### Detection Keywords

**Smart Router auto-routes on:**
- Actions: `visualize data`, `pattern recognition`, `classify`, `cluster`
- Artifacts: `dataset`, `features`, `training data`, `ml model`
- Operations: `load dataset`, `explore data`, `pca`, `tsne`
- Modes: `data viz`, `pattern recognition`, `ml visualization`

### API Reference

```python
from sub_agents.prt_visualization_wrapper import PRTVisualizationAgent

prt = PRTVisualizationAgent()

# Load dataset
result = prt.load_dataset(
    file_path="/path/to/data.csv",
    name="My Dataset",
    target_column="label"  # Optional, for supervised learning
)

# Explore
result = prt.explore_dataset(dataset_id="ds_1")

# Visualize
result = prt.visualize_dataset(
    dataset_id="ds_1",
    viz_type="correlation",  # or "distribution", "scatter", "pairplot", "box"
    columns=None  # None = all numeric columns
)

# Train classifier
result = prt.train_classifier(
    dataset_id="ds_1",
    algorithm="random_forest",  # or "svm", "logistic", "knn", "decision_tree"
    test_size=0.2
)

# Clustering
result = prt.cluster_analysis(
    dataset_id="ds_1",
    n_clusters=3,
    algorithm="kmeans"  # or "dbscan", "hierarchical"
)

# Dimensionality reduction
result = prt.dimensionality_reduction(
    dataset_id="ds_1",
    method="pca",  # or "tsne"
    n_components=2
)
```

## 4. MxGPU GPU Virtualization

### Capabilities

- **GPU Detection**: Detect AMD/NVIDIA/Intel GPUs with SR-IOV
- **SR-IOV Status**: Check VF allocation and capabilities
- **IOMMU Groups**: Map PCI devices for passthrough
- **VM Configuration**: Generate KVM/Xen configs
- **VFIO Status**: Check VFIO driver status

### Dependencies

```bash
# Linux only, no Python dependencies
# Requires: lspci, KVM/Xen, libvirt (optional)
```

### Natural Language Usage

```python
# Detect GPUs
result = orch.query("Detect available GPUs")

# Check SR-IOV status
result = orch.query(
    "Check SR-IOV status",
    pci_id="01:00.0"
)

# Get IOMMU groups
result = orch.query("Show IOMMU groups for GPU passthrough")

# Generate VM config
result = orch.query(
    "Generate VM config with GPU",
    vm_name="gaming-vm",
    gpu_pci_id="01:00.0",
    vcpus=4,
    memory_gb=8,
    hypervisor="kvm"  # or "xen"
)

# Check VFIO status
result = orch.query("Check VFIO driver status")
```

### Detection Keywords

**Smart Router auto-routes on:**
- Actions: `setup gpu`, `configure sriov`, `passthrough`, `vm gpu`
- Artifacts: `virtual machine`, `gpu`, `vf`, `pci device`
- Operations: `detect gpu`, `sriov status`, `iommu`, `vfio`, `kvm`, `xen`
- Modes: `mxgpu`, `gpu virtualization`, `sriov`, `gpu passthrough`

### API Reference

```python
from sub_agents.mxgpu_wrapper import MxGPUAgent

mxgpu = MxGPUAgent()

# Detect GPUs
result = mxgpu.detect_gpus()

# SR-IOV status
result = mxgpu.get_sriov_status(pci_id="01:00.0")

# IOMMU groups
result = mxgpu.get_iommu_groups()

# Generate VM config
result = mxgpu.generate_vm_config(
    vm_name="my-vm",
    gpu_pci_id="01:00.0",
    vcpus=4,
    memory_gb=8,
    hypervisor="kvm"
)

# VFIO status
result = mxgpu.check_vfio_status()

# Get status
status = mxgpu.get_status()
```

## Smart Router Integration

### Auto-Detection

The Smart Router automatically detects specialized queries:

```python
from smart_router import SmartRouter

router = SmartRouter()

# Geospatial
decision = router.route("Map this threat intelligence data")
# → {"model": "geospatial", "task_mode": "threat_intel", ...}

# RDKit
decision = router.route("Calculate drug-likeness for this molecule")
# → {"model": "rdkit", "task_mode": "drug_likeness", ...}

# PRT
decision = router.route("Visualize data correlations")
# → {"model": "prt", "task_mode": "visualize", ...}

# MxGPU
decision = router.route("Setup GPU passthrough for KVM")
# → {"model": "mxgpu", "task_mode": "config", ...}
```

### Detection Priority

1. **Multimodal** (images/video) → Gemini
2. **NotebookLM** (document research) → NotebookLM
3. **Geospatial** → Geospatial Agent
4. **RDKit** → RDKit Agent
5. **PRT** → PRT Agent
6. **MxGPU** → MxGPU Agent
7. **Code** → DeepSeek Coder / Qwen Coder
8. **General** → DeepSeek R1

## Installation

### Full Installation

```bash
# Geospatial dependencies
pip install geopandas folium pydeck plotly shapely pandas

# RDKit
pip install rdkit pandas numpy

# PRT (Data Visualization & ML)
pip install pandas numpy scikit-learn matplotlib seaborn plotly joblib

# MxGPU (Linux only, no Python deps)
# Ensure KVM/Xen and IOMMU are configured

# Minimal installation (just pandas/numpy)
pip install pandas numpy
```

### Verification

```python
from unified_orchestrator import UnifiedAIOrchestrator

orch = UnifiedAIOrchestrator()

# Check status
import json
status = orch.get_status()
print(json.dumps(status['specialized_agents'], indent=2))

# Output:
# {
#   "geospatial": true/false,
#   "rdkit": true/false,
#   "prt": true/false,
#   "mxgpu": true/false
# }
```

## Examples

### Example 1: OSINT Threat Intelligence Mapping

```python
from unified_orchestrator import UnifiedAIOrchestrator

orch = UnifiedAIOrchestrator()

# Load threat data
result = orch.query(
    "Load geospatial threat data",
    file_path="/data/apt_campaign.geojson",
    dataset_name="APT28 Campaign 2024"
)

dataset_id = result['dataset_id']

# Create threat map
result = orch.query(
    "Create threat intelligence map",
    dataset_ids=[dataset_id],
    map_type="folium",
    title="APT28 Infrastructure - 2024",
    style="dark"
)

print(f"Map saved to: {result['file_path']}")

# Analyze hotspots
result = orch.query(
    "Analyze threat hotspots",
    dataset_id=dataset_id,
    analysis_type="hotspot"
)

print(f"Top hotspots: {result['hotspots']}")
```

### Example 2: Drug Discovery Workflow

```python
# Parse candidate molecules
candidates = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
]

mol_ids = []
for i, smiles in enumerate(candidates):
    result = orch.query(
        "Parse molecular structure",
        structure=smiles,
        format="smiles",
        name=f"Candidate_{i+1}"
    )
    mol_ids.append(result['molecule_id'])

# Analyze drug-likeness
for mol_id in mol_ids:
    result = orch.query(
        "Analyze drug-likeness",
        mol_id=mol_id
    )
    print(f"{mol_id}: {result['overall_drug_likeness']}")

# Find similar compounds
result = orch.query(
    "Find similar compounds",
    query_mol_id=mol_ids[0],
    target_mol_ids=mol_ids[1:],
    fp_type="morgan",
    metric="tanimoto"
)

print(f"Similarity results: {result['results']}")
```

### Example 3: ML Classification Pipeline

```python
# Load dataset
result = orch.query(
    "Load dataset",
    file_path="/data/customer_data.csv",
    name="Customer Segmentation",
    target_column="purchased"
)

dataset_id = result['dataset_id']

# Explore
result = orch.query(
    "Explore dataset statistics",
    dataset_id=dataset_id
)

# Visualize correlations
result = orch.query(
    "Visualize correlations",
    dataset_id=dataset_id,
    viz_type="correlation"
)

# Train classifier
result = orch.query(
    "Train random forest classifier",
    dataset_id=dataset_id,
    algorithm="random_forest",
    test_size=0.2
)

print(f"Model accuracy: {result['accuracy']}")
print(f"Model ID: {result['model_id']}")
```

### Example 4: GPU Virtualization Setup

```python
# Detect GPUs
result = orch.query("Detect available GPUs")
gpus = result['gpus']

if gpus:
    gpu_pci_id = gpus[0]['pci_id']

    # Check SR-IOV
    result = orch.query(
        "Check SR-IOV status",
        pci_id=gpu_pci_id
    )

    # Generate KVM config
    result = orch.query(
        "Generate KVM VM config with GPU",
        vm_name="gaming-vm",
        gpu_pci_id=gpu_pci_id,
        vcpus=8,
        memory_gb=16,
        hypervisor="kvm"
    )

    print(f"Config saved to: {result['config_file']}")
```

### Example 5: NMDA Antidepressant Screening

```python
# Parse ketamine analog
result = orch.query(
    "Parse molecule for NMDA analysis",
    structure="CC(=O)C(c1ccccc1Cl)N(C)C",
    name="Ketamine"
)

mol_id = result['mol_id']

# Analyze NMDA activity
result = orch.query(
    "Analyze NMDA antidepressant potential",
    mol_id=mol_id
)

print(f"NMDA Activity Score: {result['nmda_activity_score']}/10")
print(f"BBB Penetration: {result['bbb_penetration']['prediction']}")

# Compare with known antidepressants
result = orch.query(
    "Compare with ketamine and esketamine",
    mol_id=mol_id
)

print(f"Most similar to: {result['most_similar_drug']}")
print(f"Novelty score: {result['novelty_score']}/10")
```

### Example 6: NPS Abuse Potential Prediction

```python
# Parse potential designer drug
result = orch.query(
    "Parse NPS for threat analysis",
    structure="CCN(CC)C(=O)C1CN(C)CCc2ccccc21",
    name="Suspected_Fentanyl_Analog"
)

mol_id = result['mol_id']

# Classify as NPS
result = orch.query(
    "Classify novel psychoactive substance",
    mol_id=mol_id
)

if result['is_nps']:
    print(f"⚠️  WARNING: Classified as {result['nps_class']}")
    print(f"DEA Schedule: {result['dea_schedule']}")

    # Predict abuse potential (comprehensive 12-hour analysis)
    result = orch.query(
        "Comprehensive abuse potential analysis",
        mol_id=mol_id,
        comprehensive=True
    )

    print(f"Abuse Potential Score: {result['abuse_potential_score']}/10")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Lethality Risk: {result['lethality_assessment']['lethality_score']}/10")

    # Get antidote recommendations
    if result['antidote_recommendations']['primary_antidotes']:
        print("\nAntidotes:")
        for antidote in result['antidote_recommendations']['primary_antidotes']:
            print(f"  • {antidote}")

    # Regulatory recommendations
    if result.get('recommendations'):
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")
```

## 5. NMDA Agonist Antidepressant Analysis

### Capabilities

- **NMDA Receptor Activity Prediction** - Structural similarity to known NMDA modulators
- **Blood-Brain Barrier (BBB) Analysis** - Lipinski-based BBB penetration prediction
- **Comparison with Known Antidepressants** - Similarity to ketamine, esketamine, memantine
- **Antidepressant Potential Scoring** - 0-10 scale based on multiple factors
- **Safety Warnings** - Neurotoxicity and addiction potential assessment
- **Batch Screening** - Analyze multiple compounds for top candidates

### Dependencies

```bash
# Uses RDKit (already installed for cheminformatics)
pip install rdkit
```

### Natural Language CLI

```bash
# Parse molecules
python3 nmda_cli.py 'parse "CC(=O)C(c1ccccc1Cl)N(C)C" as Ketamine'

# Analyze NMDA activity
python3 nmda_cli.py "analyze nmda activity for mol_1"

# Blood-Brain Barrier prediction
python3 nmda_cli.py "check bbb penetration for mol_1"

# Compare with known antidepressants
python3 nmda_cli.py "compare mol_1 with known antidepressants"

# Comprehensive analysis
python3 nmda_cli.py "comprehensive analysis of mol_1"

# Batch screening
python3 nmda_cli.py "batch analyze mol_1 mol_2 mol_3"

# List molecules
python3 nmda_cli.py "list molecules"
```

### API Reference

```python
from sub_agents.nmda_agonist_analyzer import NMDAAgonistAnalyzer

analyzer = NMDAAgonistAnalyzer()

# Parse molecule (uses RDKit)
result = analyzer.rdkit_agent.parse_molecule(
    structure="CC(=O)C(c1ccccc1Cl)N(C)C",
    format="smiles",
    name="Ketamine"
)
mol_id = result['mol_id']

# Analyze NMDA activity
result = analyzer.analyze_nmda_activity(mol_id=mol_id)
# Returns: nmda_activity_score, likely_mechanism, bbb_penetration, warnings

# Blood-Brain Barrier prediction
result = analyzer.predict_bbb_penetration(mol_id=mol_id)
# Returns: bbb_score, prediction ('High BBB+', 'Moderate BBB+', 'Low BBB+', 'BBB-')

# Compare with known antidepressants
result = analyzer.compare_with_known_antidepressants(mol_id=mol_id)
# Returns: similarity_scores (ketamine, esketamine, memantine), most_similar_drug, novelty_score

# Comprehensive analysis
result = analyzer.comprehensive_analysis(mol_id=mol_id)
# Returns: All of the above + overall assessment

# Batch screening
result = analyzer.batch_analysis(
    mol_ids=["mol_1", "mol_2", "mol_3"],
    output_dir="/path/to/results"
)
# Returns: summary with top candidates ranked by potential

# Get status
status = analyzer.get_status()
# Returns: available, loaded_molecules, known_antidepressants, capabilities
```

### Known Reference Compounds

The analyzer compares novel compounds against:
- **Ketamine** (Ketalar) - NMDA antagonist, rapid-acting antidepressant
- **Esketamine** (Spravato) - S-enantiomer of ketamine, FDA-approved for TRD
- **Memantine** (Namenda) - NMDA antagonist, used in Alzheimer's disease

### Use Cases

1. **Novel Antidepressant Discovery** - Screen novel compounds for NMDA activity
2. **Ketamine Analog Research** - Identify safer/more effective ketamine derivatives
3. **Treatment-Resistant Depression** - Find rapid-acting antidepressant candidates
4. **Pharmaceutical Research** - BBB-permeable CNS drug development
5. **Academic Research** - NMDA receptor pharmacology studies

## 6. NPS Abuse Potential Analysis

### Capabilities

- **NPS Classification** - 20+ chemical classes (synthetic cannabinoids, cathinones, fentanyl analogs, etc.)
- **Abuse Potential Scoring** - 0-10 scale prediction for recreational use likelihood
- **Receptor Binding Prediction** - 6 neurotransmitter systems (opioid, dopamine, serotonin, etc.)
- **Neurotoxicity Assessment** - CNS damage risk evaluation
- **Lethality Risk** - LD50 estimation and overdose potential
- **Antidote Recommendations** - Emergency treatment protocols (naloxone, flumazenil, etc.)
- **Dark Web Proliferation Prediction** - Likelihood of emerging as designer drug
- **DEA Scheduling Recommendations** - Regulatory classification guidance
- **Comprehensive 12-Hour Analysis Mode** - Deep molecular dynamics for critical assessments
- **Batch Screening** - Large-scale threat intelligence (1M+ compounds)

### Dependencies

```bash
# Uses RDKit (already installed for cheminformatics)
pip install rdkit
```

### Natural Language CLI

```bash
# Parse molecules
python3 nps_cli.py 'parse "CCN(CC)C(=O)C1CN(C)CCc2ccccc21" as Fentanyl'

# Classify as NPS
python3 nps_cli.py "classify mol_1"

# Predict abuse potential (standard mode)
python3 nps_cli.py "predict abuse potential for mol_1"

# Comprehensive 12-hour analysis
python3 nps_cli.py "comprehensive abuse analysis of mol_1"

# Receptor binding prediction
python3 nps_cli.py "predict receptor binding for mol_1"

# Antidote recommendations
python3 nps_cli.py "recommend antidote for mol_1"

# Batch screening
python3 nps_cli.py "batch screen mol_1 mol_2 mol_3"

# List molecules
python3 nps_cli.py "list molecules"
```

### API Reference

```python
from sub_agents.nps_abuse_potential_analyzer import NPSAbusePotentialAnalyzer

analyzer = NPSAbusePotentialAnalyzer(verbose=True)

# Parse molecule (uses RDKit)
result = analyzer.rdkit_agent.parse_molecule(
    structure="CCN(CC)C(=O)C1CN(C)CCc2ccccc21",
    format="smiles",
    name="Fentanyl"
)
mol_id = result['mol_id']

# Classify as NPS
result = analyzer.classify_nps(mol_id=mol_id)
# Returns: is_nps, nps_class, controlled_substance, dea_schedule, matched_patterns

# Predict abuse potential
result = analyzer.predict_abuse_potential(
    mol_id=mol_id,
    comprehensive=False  # Set True for 12-hour deep analysis
)
# Returns: abuse_potential_score, risk_category, nps_classification,
#          similarity_analysis, receptor_binding, reinforcement_mechanisms,
#          neurotoxicity_assessment, lethality_assessment, antidote_recommendations,
#          proliferation_prediction, warnings, recommendations

# Receptor binding prediction
result = analyzer.predict_receptor_binding(mol_id=mol_id)
# Returns: receptor_predictions (opioid, dopamine, serotonin, nmda, gaba, cannabinoid)

# Batch screening (large-scale threat intelligence)
result = analyzer.batch_screening(
    mol_ids=["mol_1", "mol_2", ..., "mol_1000000"],
    output_dir="/path/to/threat_intel"
)
# Returns: total_screened, high_risk_count, medium_risk_count, low_risk_count,
#          high_risk_substances, processing_time

# Get status
status = analyzer.get_status()
# Returns: available, nps_classes, reference_drugs, capabilities, receptor_systems
```

### Supported NPS Classes

- **Synthetic Cannabinoids** - JWH-018, AB-FUBINACA, 5F-ADB, etc.
- **Synthetic Cathinones** - Mephedrone, MDPV, Alpha-PVP, flakka
- **Fentanyl Analogs** - Carfentanil, acetylfentanyl, furanylfentanyl
- **Benzodiazepine Analogs** - Etizolam, flualprazolam, clonazolam
- **NBOMe Compounds** - 25I-NBOMe, 25C-NBOMe, 25B-NBOMe
- **2C-x Psychedelics** - 2C-B, 2C-I, 2C-E, 2C-T-7
- **Tryptamines** - DMT, 5-MeO-DMT, DPT, 4-AcO-DMT
- **Phenethylamines** - MDMA analogs, DOx compounds
- **Designer Opioids** - U-47700, MT-45, AH-7921
- **And 10+ more classes**

### Receptor Systems Analyzed

1. **Opioid Receptors** (μ, κ, δ) - Overdose risk, respiratory depression
2. **Dopamine Receptors** - Addiction potential, euphoria
3. **Serotonin Receptors** (5-HT) - Psychedelic effects, serotonin syndrome
4. **NMDA/Glutamate** - Dissociative effects, neurotoxicity
5. **GABA Receptors** - Sedation, respiratory depression
6. **Cannabinoid Receptors** (CB1, CB2) - Synthetic cannabinoid effects

### PROACTIVE Capabilities

This analyzer is designed for **PROACTIVE** threat intelligence:

1. **Predict Not-Yet-Synthesized Substances** - Analyze hypothetical structures before they emerge
2. **Dark Web Forecasting** - Estimate likelihood of proliferation on underground markets
3. **Early Antidote Development** - Recommend countermeasures before substances appear
4. **Regulatory Pre-emption** - Generate DEA scheduling recommendations for analog act application
5. **Large-Scale Screening** - Batch analyze 1M+ compounds to identify emerging threats

### Abuse Potential Scoring

**0-3**: Low risk (caffeine, nicotine-like)
**4-6**: Moderate risk (prescription stimulants, opioids)
**7-8**: High risk (cocaine, heroin-like)
**9-10**: Extreme risk (fentanyl, carfentanil-like)

### Use Cases

1. **Law Enforcement Threat Intelligence** - PROACTIVE identification of emerging designer drugs
2. **Regulatory Agency Decision Support** - DEA/FDA scheduling recommendations
3. **Emergency Medicine Preparedness** - Antidote stockpiling for predicted substances
4. **Pharmaceutical Safety Assessment** - Screen novel compounds for abuse liability
5. **Academic Research** - Neuropharmacology and addiction science
6. **Border/Customs Screening** - Identify suspicious chemical structures in shipments

### TEMPEST Compliance

Both NMDA and NPS analyzers are:
- **100% Local** - No cloud dependencies, all RDKit computations
- **Air-gapped Compatible** - Zero network communication required
- **Classified-Safe** - Suitable for SCIF deployments
- **EM Emissions** - Minimal (CPU-bound molecular calculations only)
- **ITAR Compliant** - No export-controlled algorithms

### Ethical & Legal Notice

**⚠️  CRITICAL: Authorized Use Only**

The NPS Abuse Potential Analyzer is designed for:
- ✅ Law enforcement threat intelligence
- ✅ Pharmaceutical safety research
- ✅ Emergency medicine preparedness
- ✅ Regulatory agency decision support
- ✅ Academic neuropharmacology research

**NOT** for:
- ❌ Illicit drug synthesis guidance
- ❌ Recreational drug design
- ❌ Circumventing controlled substance laws
- ❌ Dark web marketplace guidance

**Results are PREDICTIONS, not clinical validation**. All high-risk compounds require wet-lab validation before regulatory action. This tool is designed to **PREVENT HARM**, not enable it.

## Troubleshooting

### Geospatial

```
Error: GeoPandas not installed
```
**Solution**: `pip install geopandas folium`

### RDKit

```
Error: RDKit not installed
```
**Solution**: `pip install rdkit`

### PRT

```
Error: Pandas/NumPy not installed
```
**Solution**: `pip install pandas numpy scikit-learn matplotlib seaborn`

### MxGPU

```
Error: GPU detection requires Linux
```
**Solution**: MxGPU only works on Linux systems

## Performance

| Agent | Operation | Typical Time |
|-------|-----------|--------------|
| Geospatial | Load GeoJSON (1K features) | ~2-3s |
| Geospatial | Create map (5K features) | ~5-8s |
| RDKit | Parse molecule | ~100ms |
| RDKit | Calculate descriptors | ~200ms |
| RDKit | Generate fingerprint | ~50ms |
| PRT | Load CSV (10K rows) | ~1-2s |
| PRT | Train classifier | ~5-30s |
| PRT | Visualize | ~2-5s |
| MxGPU | Detect GPUs | ~200ms |
| MxGPU | Generate config | ~50ms |
| NMDA | Analyze NMDA activity | ~1-2s |
| NMDA | BBB prediction | ~500ms |
| NMDA | Comprehensive analysis | ~3-5s |
| NPS | Classify NPS | ~1-2s |
| NPS | Predict abuse potential | ~2-5s |
| NPS | Comprehensive (12-hour mode) | Up to 12 hours |
| NPS | Batch screening (1M compounds) | ~24 hours |

## Credits

- **Geospatial**: Inspired by OpenSphere (NGA)
- **Cheminformatics**: Powered by RDKit
- **Data Viz**: Inspired by MATLAB PRT
- **GPU Virtualization**: Inspired by AMD MxGPU
- **NMDA Analysis**: Novel implementation using RDKit computational chemistry
- **NPS Analysis**: PROACTIVE threat intelligence framework
- **Implementation**: LAT5150DRVMIL AI Platform

## License

Part of the LAT5150DRVMIL project.

