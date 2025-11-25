# Data Management Guide
## LAT5150DRVMIL AI Engine

Complete guide to data management, version control, validation, and governance.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Version Control (DVC)](#data-version-control-dvc)
3. [Data Validation](#data-validation)
4. [Data Policies](#data-policies)
5. [Data Categories](#data-categories)
6. [Quick Start](#quick-start)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The LAT5150DRVMIL data management system provides:

✅ **Version Control** - Track datasets with DVC (Data Version Control)
✅ **Automated Validation** - Ensure data integrity and quality
✅ **Clear Policies** - Governance for usage, retention, and security
✅ **Data Classification** - Separate sample, dev, and production data
✅ **Reproducibility** - Track data versions for experiment reproducibility

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **DVC** | Data version control | `.dvc/` config, `*.dvc` tracking files |
| **Validator** | Automated data validation | `data_management/validators/data_validator.py` |
| **Policies** | Data governance policies | `data_management/policies/DATA_POLICIES.md` |
| **Sample Data** | Development/test datasets | `sample_data/` |

---

## Data Version Control (DVC)

### What is DVC?

DVC (Data Version Control) is Git for data. It tracks changes to datasets,
models, and large files without bloating your Git repository.

### Why DVC?

- ✅ **Reproducibility** - Track exact versions of data used in experiments
- ✅ **Collaboration** - Share datasets across team
- ✅ **Storage Efficiency** - Store large files externally
- ✅ **Versioning** - Track changes to datasets over time

### Installation

```bash
# Install DVC
pip install dvc

# Or via requirements
make install-dev
```

### Quick Start

#### 1. Initialize DVC (Already Done)

The project is already configured with DVC:
- Configuration: `.dvc/config`
- Ignore patterns: `.dvcignore`
- Local storage: `.dvc/cache/`

#### 2. Track a Dataset

```bash
# Track a new dataset
dvc add sample_data/my_dataset.csv

# This creates my_dataset.csv.dvc (tracked in Git)
# and adds my_dataset.csv to .gitignore

# Commit the .dvc file
git add sample_data/my_dataset.csv.dvc sample_data/.gitignore
git commit -m "Track my_dataset with DVC"
```

#### 3. Update a Dataset

```bash
# Update the file
echo "new data" >> sample_data/my_dataset.csv

# Update DVC tracking
dvc add sample_data/my_dataset.csv

# Commit the change
git add sample_data/my_dataset.csv.dvc
git commit -m "Update my_dataset"
```

#### 4. Retrieve a Dataset

```bash
# Get the latest version
dvc pull sample_data/my_dataset.csv

# Get a specific version
git checkout <commit-hash> sample_data/my_dataset.csv.dvc
dvc checkout sample_data/my_dataset.csv.dvc
```

### Cloud Storage (Optional)

Configure remote storage for team collaboration:

#### AWS S3

```bash
# Configure S3 remote
dvc remote add -d s3 s3://your-bucket/dvc-storage
dvc remote modify s3 region us-west-2

# Push data to S3
dvc push

# Pull data from S3
dvc pull
```

#### Google Cloud Storage

```bash
# Configure GCS remote
dvc remote add -d gcs gs://your-bucket/dvc-storage

# Push/pull
dvc push
dvc pull
```

#### Azure Blob Storage

```bash
# Configure Azure remote
dvc remote add -d azure azure://your-container/dvc-storage

# Push/pull
dvc push
dvc pull
```

### Common DVC Commands

```bash
# Add/track file
dvc add <file>

# Check status
dvc status

# Push to remote
dvc push

# Pull from remote
dvc pull

# List tracked files
dvc list . --dvc-only

# Show file info
dvc info <file.dvc>

# Compare versions
dvc diff

# Reproduce pipeline (if using dvc run)
dvc repro
```

### Makefile Integration

```bash
# Validate all tracked data
make validate-data

# DVC status check
dvc status

# Push data to remote
dvc push
```

---

## Data Validation

### Overview

Automated validation ensures data integrity, format compliance, and quality standards.

### Features

- ✅ **Schema Validation** - Check structure and types
- ✅ **Value Validation** - Range checks, patterns, formats
- ✅ **Missing Value Detection** - Identify incomplete data
- ✅ **Duplicate Detection** - Find duplicate records
- ✅ **Statistical Anomalies** - Detect outliers
- ✅ **Custom Rules** - Add domain-specific validations

### Quick Start

#### 1. Basic Validation

```python
from data_management.validators.data_validator import DataValidator

# Create validator
validator = DataValidator()

# Validate CSV file
report = validator.validate_file("sample_data/california_housing_test.csv")

# Print results
validator.print_report(report)

# Export to JSON
validator.export_report(report, "validation_reports/housing_report.json")
```

#### 2. Validation with Schema

```python
# Define schema
schema = {
    'median_income': {'type': 'number', 'min': 0, 'max': 15},
    'median_house_value': {'type': 'number', 'min': 0},
    'latitude': {'type': 'number', 'min': -90, 'max': 90},
    'longitude': {'type': 'number', 'min': -180, 'max': 180}
}

# Validate with schema
report = validator.validate_file(
    "sample_data/california_housing_test.csv",
    schema=schema
)
```

#### 3. Custom Validation Rules

```python
from data_management.validators.data_validator import ValidationRule

# Add custom rule
validator.add_rule(ValidationRule(
    name="valid_email",
    description="Check if email format is valid",
    validator=lambda v: '@' in v and '.' in v.split('@')[1],
    severity="error",
    applies_to=["email"]  # Only check 'email' column
))

# Validate with custom rules
report = validator.validate_file("sample_data/users.csv")
```

### Validation Output

```
================================================================================
  DATA VALIDATION REPORT: sample_data/california_housing_test.csv
================================================================================

📊 File Info:
   Type: .csv
   Size: 1,234 bytes (1.21 KB)
   Hash: a1b2c3d4e5f6...
   Dimensions: 10 rows × 9 columns

✓ PASSED

📋 Validation Summary:
   Errors: 0
   Warnings: 2
   Total Issues: 2
   Validation Time: 0.045s

⚠️  Issues Found:
   ⚠️  [WARNING] Column name 'medianIncome' doesn't follow naming conventions
      Location: header:medianIncome
   ⚠️  [WARNING] Duplicate row found
      Location: row:5

================================================================================
```

### Command Line Validation

```bash
# Validate single file
python3 data_management/validators/data_validator.py \
    --file sample_data/test.csv \
    --schema schemas/test_schema.json \
    --output validation_reports/test_report.json

# Validate all files in directory
for file in sample_data/*.csv; do
    python3 data_management/validators/data_validator.py --file "$file"
done
```

### CI/CD Integration

Add to `.github/workflows/data-validation.yml`:

```yaml
name: Data Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Validate data
        run: |
          python3 data_management/validators/data_validator.py \
            --file sample_data/*.csv \
            --fail-on-error
```

---

## Data Policies

### Overview

Comprehensive data governance policies covering:
- Data classification and sensitivity
- Lifecycle management (acquisition → deletion)
- Usage policies and restrictions
- Security and privacy requirements
- Quality standards and SLAs
- Retention and deletion schedules
- Compliance and audit requirements

### Key Policies

📋 **Full Documentation:** `data_management/policies/DATA_POLICIES.md`

#### Data Classification

| Category | Purpose | Sensitivity | Version Control |
|----------|---------|-------------|-----------------|
| **Sample Data** | Dev/test | Public | Git + DVC |
| **Development Data** | Dev environment | Internal | DVC only |
| **Production Data** | Live system | Confidential | Not tracked |
| **Training Data** | Model training | Varies | DVC |
| **Cache Data** | Performance | Varies | Not tracked |

#### Data Lifecycle

```
Acquisition → Processing → Active Use → Archive → Deletion
```

#### Retention Periods

| Data Type | Retention | Rationale |
|-----------|-----------|-----------|
| Sample Data | Permanent | Part of codebase |
| Dev Data | 90 days | Cost optimization |
| Production Logs | 90 days | Debugging |
| Training Data | Permanent (versioned) | Reproducibility |
| Cache Data | 1hr - 7 days (TTL) | Performance |

#### Security Requirements

- **Encryption at Rest:** AES-256 for production data
- **Encryption in Transit:** TLS 1.3+ for all transfers
- **Access Control:** RBAC with principle of least privilege
- **Audit Logging:** 1 year retention for access logs

---

## Data Categories

### 1. Sample Data (`sample_data/`)

**Purpose:** Local development, testing, demonstrations

**Characteristics:**
- ✓ Small, curated datasets
- ✓ Non-sensitive, public data
- ✓ Tracked in Git and DVC
- ✓ Included in development builds
- ✓ Used in documentation and examples

**Examples:**
- `california_housing_test.csv` - Housing price data (10 records)
- `anscombe.json` - Statistical demo data (Anscombe's quartet)
- `mnist_test.csv` - Handwritten digit samples (100 images)

**Usage:**
```python
# Safe to use anywhere
import pandas as pd
df = pd.read_csv("sample_data/california_housing_test.csv")
```

### 2. Development Data (`dev_data/`)

**Purpose:** Development environment testing

**Characteristics:**
- ⚠️ Larger datasets, anonymized
- ⚠️ Internal use only
- ✓ Tracked in DVC (not Git)
- ✗ Not in production builds
- ✓ Periodic refresh from production (anonymized)

**Usage:**
```python
# Use for development, not production
df = pd.read_csv("dev_data/test_dataset.csv")
```

### 3. Production Data (`production_data/`)

**Purpose:** Live system operations

**Characteristics:**
- 🔒 Highly sensitive
- 🔒 Encrypted at rest and in transit
- ✗ Never tracked in version control
- ✗ Never included in builds
- ✓ Strict access controls

**Usage:**
```python
# Only in production environment
# with proper authentication
import os
data_path = os.environ['PRODUCTION_DATA_PATH']
df = load_encrypted_data(data_path)
```

### 4. Training Data (`training_data/`)

**Purpose:** Model training and fine-tuning

**Characteristics:**
- ✓ Versioned with DVC
- ✓ Permanent retention
- ⚠️ Sensitivity varies (must be specified)
- ✗ Not in builds
- ✓ Documented provenance

**Usage:**
```python
# Load specific version
import dvc.api
with dvc.api.open(
    'training_data/model_v2.csv',
    rev='v2.1.0'
) as f:
    df = pd.read_csv(f)
```

### 5. Cache Data (`~/.cache/lat5150drvmil/`)

**Purpose:** Performance optimization

**Characteristics:**
- ⚡ Temporary, auto-managed
- ✓ TTL-based expiration
- ⚠️ Inherits sensitivity from source
- ✗ Not tracked
- ✓ Automatic cleanup

**Usage:**
```python
from performance import cached

@cached(cache_name="api_responses", ttl=3600)
def fetch_data():
    # Automatically cached
    return api.get_data()
```

---

## Quick Start

### For New Developers

```bash
# 1. Clone repository
git clone <repository>
cd LAT5150DRVMIL

# 2. Install dependencies (includes DVC)
make install-dev

# 3. Pull data from DVC
dvc pull

# 4. Validate sample data
python3 data_management/validators/data_validator.py \
    --file sample_data/california_housing_test.csv

# 5. Start developing!
```

### Adding New Data

```bash
# 1. Add file to appropriate directory
cp my_dataset.csv sample_data/

# 2. Validate it
python3 data_management/validators/data_validator.py \
    --file sample_data/my_dataset.csv

# 3. Track with DVC
dvc add sample_data/my_dataset.csv

# 4. Commit
git add sample_data/my_dataset.csv.dvc sample_data/.gitignore
git commit -m "Add my_dataset"

# 5. Push to remote (optional)
dvc push
git push
```

### Using Data in Code

```python
import pandas as pd
from data_management.validators.data_validator import DataValidator

# 1. Load data
df = pd.read_csv("sample_data/california_housing_test.csv")

# 2. Validate (optional but recommended)
validator = DataValidator()
report = validator.validate_file("sample_data/california_housing_test.csv")

if not report.passed:
    print(f"⚠️  Data validation failed with {report.error_count} errors")
    validator.print_report(report)
    raise ValueError("Data validation failed")

# 3. Use data
print(f"✓ Data validated: {len(df)} rows")
# ... your code ...
```

---

## Best Practices

### 1. Data Version Control

✅ **DO:**
- Track all datasets with DVC
- Use semantic versioning for data (v1.0.0, v1.1.0)
- Document data changes in commit messages
- Tag important data versions (`git tag data-v1.0.0`)
- Use remote storage for team collaboration

✗ **DON'T:**
- Commit large files directly to Git
- Skip DVC tracking for "small" files (use DVC anyway)
- Forget to `dvc push` after `dvc add`
- Delete .dvc files (you'll lose tracking)

### 2. Data Validation

✅ **DO:**
- Validate data on ingestion
- Create schemas for known formats
- Run validation in CI/CD
- Keep validation reports for audit
- Fix validation errors before using data

✗ **DON'T:**
- Skip validation for "trusted" sources
- Ignore warnings (investigate them)
- Use data with validation errors
- Validate only once (re-validate periodically)

### 3. Data Security

✅ **DO:**
- Classify data by sensitivity
- Encrypt production data
- Use access controls
- Anonymize data for development
- Audit data access regularly

✗ **DON'T:**
- Mix production and development data
- Share sensitive data via insecure channels
- Include production data in version control
- Grant broad access "just in case"

### 4. Data Quality

✅ **DO:**
- Document data sources and lineage
- Track data quality metrics
- Set quality SLAs
- Monitor data drift
- Automate quality checks

✗ **DON'T:**
- Assume data quality
- Skip documentation
- Ignore data quality degradation
- Use outdated data without checking

---

## Troubleshooting

### DVC Issues

**Problem:** `dvc pull` fails

```bash
# Check DVC status
dvc status

# Check remote configuration
dvc remote list

# Try pulling specific file
dvc pull sample_data/my_dataset.csv.dvc

# Check cache
dvc cache dir
ls -la .dvc/cache/
```

**Problem:** `.dvc` file out of sync

```bash
# Checkout correct version
git checkout <commit> sample_data/my_dataset.csv.dvc
dvc checkout sample_data/my_dataset.csv.dvc

# Or re-add file
dvc add sample_data/my_dataset.csv
```

### Validation Issues

**Problem:** Validation fails with many errors

```bash
# Get detailed report
python3 data_management/validators/data_validator.py \
    --file sample_data/my_dataset.csv \
    --verbose \
    --output report.json

# Check the report
cat report.json | jq '.issues[] | select(.severity == "error")'
```

**Problem:** Custom rule not working

```python
# Debug rule
rule = ValidationRule(
    name="test_rule",
    description="Test",
    validator=lambda v: print(f"Checking: {v}") or True  # Debug print
)
validator.add_rule(rule)
```

### Data Access Issues

**Problem:** Can't find data file

```bash
# Check file exists
ls -la sample_data/

# Check DVC tracking
dvc list . --dvc-only

# Pull from DVC
dvc pull
```

**Problem:** Permission denied

```bash
# Check file permissions
ls -la sample_data/my_dataset.csv

# Fix permissions
chmod 644 sample_data/my_dataset.csv
```

---

## Advanced Topics

### Automated Data Pipelines

```python
# Example: Automated validation pipeline
import os
from pathlib import Path
from data_management.validators.data_validator import DataValidator

def validate_directory(directory: str, output_dir: str):
    """Validate all CSV files in directory"""
    validator = DataValidator()
    results = []

    for file_path in Path(directory).glob("*.csv"):
        print(f"Validating {file_path}...")
        report = validator.validate_file(str(file_path))

        # Export report
        report_name = f"{file_path.stem}_report.json"
        report_path = Path(output_dir) / report_name
        validator.export_report(report, str(report_path))

        results.append({
            'file': str(file_path),
            'passed': report.passed,
            'errors': report.error_count
        })

    return results

# Run pipeline
results = validate_directory("sample_data", "validation_reports")
print(f"Validated {len(results)} files")
print(f"Passed: {sum(1 for r in results if r['passed'])}")
```

### Data Versioning Strategy

```bash
# Strategy 1: Git tags for data versions
dvc add training_data/model_data.csv
git add training_data/model_data.csv.dvc
git commit -m "Update training data"
git tag data-v1.2.0
git push origin data-v1.2.0

# Strategy 2: Branches for data experiments
git checkout -b experiment/new-data-source
dvc add training_data/experimental_data.csv
git add training_data/experimental_data.csv.dvc
git commit -m "Add experimental data source"

# Strategy 3: DVC experiments (for ML workflows)
dvc exp run --set-param data_version=v2
```

### Data Quality Monitoring

```python
# Monitor data quality over time
import pandas as pd
from datetime import datetime

def track_data_quality(file_path: str, metrics_file: str):
    """Track quality metrics over time"""
    validator = DataValidator()
    report = validator.validate_file(file_path)

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'file': file_path,
        'passed': report.passed,
        'errors': report.error_count,
        'warnings': report.warning_count,
        'rows': report.row_count,
        'file_size': report.file_size
    }

    # Append to metrics file
    df = pd.DataFrame([metrics])
    if os.path.exists(metrics_file):
        df_existing = pd.read_csv(metrics_file)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(metrics_file, index=False)
    return metrics
```

---

## Summary

The LAT5150DRVMIL data management system provides:

✅ **Version Control** - DVC for dataset tracking and reproducibility
✅ **Validation** - Automated quality checks and schema validation
✅ **Governance** - Clear policies for usage, security, and compliance
✅ **Classification** - Separate sample, dev, and production data
✅ **Integration** - Works with Git, CI/CD, and build system

### Quick Commands

```bash
# DVC
dvc add <file>              # Track file
dvc pull                    # Get data
dvc push                    # Share data

# Validation
python3 data_management/validators/data_validator.py --file <file>

# Makefile
make validate-data          # Validate all data
make help                   # See all commands
```

### Documentation

- **This Guide:** Complete data management overview
- **Policies:** `data_management/policies/DATA_POLICIES.md`
- **DVC Docs:** https://dvc.org/doc
- **Validator API:** See `data_management/validators/data_validator.py`

---

## Contact

**Issues:** https://github.com/LAT5150DRVMIL/issues
**Data Governance:** data-governance@lat5150drvmil.org
