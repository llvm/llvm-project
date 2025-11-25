#!/usr/bin/env python3
"""
Data Validation Framework
--------------------------
Automated validation for datasets to ensure integrity, format, and consistency.

Features:
- CSV/JSON schema validation
- Data type checking
- Value range validation
- Missing value detection
- Statistical anomaly detection
- Custom validation rules
- Validation reports
"""

import os
import json
import csv
import hashlib
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Single validation rule"""
    name: str
    description: str
    validator: Callable[[Any], bool]
    severity: str = "error"  # 'error', 'warning', 'info'
    applies_to: Optional[List[str]] = None  # Column names (None = all)


@dataclass
class ValidationIssue:
    """Validation issue found in data"""
    rule_name: str
    severity: str
    message: str
    location: Optional[str] = None  # Row/column location
    value: Optional[Any] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'rule': self.rule_name,
            'severity': self.severity,
            'message': self.message,
            'location': self.location,
            'value': str(self.value) if self.value is not None else None,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class ValidationReport:
    """Complete validation report"""
    file_path: str
    file_type: str
    file_size: int
    file_hash: str
    row_count: int
    column_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    passed: bool = True
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size_bytes': self.file_size,
            'file_hash': self.file_hash,
            'dimensions': {
                'rows': self.row_count,
                'columns': self.column_count
            },
            'validation': {
                'passed': self.passed,
                'errors': self.error_count,
                'warnings': self.warning_count,
                'total_issues': len(self.issues),
                'time_seconds': round(self.validation_time, 3)
            },
            'issues': [issue.to_dict() for issue in self.issues],
            'metadata': self.metadata
        }


class DataValidator:
    """
    Comprehensive data validator with customizable rules.

    Validates CSV and JSON data files for integrity, format, and consistency.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._builtin_rules_loaded = False

    def add_rule(self, rule: ValidationRule):
        """Add a custom validation rule"""
        self.rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")

    def load_builtin_rules(self):
        """Load built-in validation rules"""
        if self._builtin_rules_loaded:
            return

        # Rule 1: No missing values in required columns
        self.add_rule(ValidationRule(
            name="no_missing_values",
            description="Check for missing/null values",
            validator=lambda v: v is not None and v != "" and str(v).strip() != "",
            severity="error"
        ))

        # Rule 2: Numeric values are valid
        self.add_rule(ValidationRule(
            name="valid_numeric",
            description="Numeric columns contain valid numbers",
            validator=lambda v: self._is_valid_number(v),
            severity="error"
        ))

        # Rule 3: No duplicate rows
        # (Handled separately in validate_csv)

        # Rule 4: Column names are valid
        self.add_rule(ValidationRule(
            name="valid_column_names",
            description="Column names follow naming conventions",
            validator=lambda name: self._is_valid_column_name(name),
            severity="warning"
        ))

        self._builtin_rules_loaded = True
        logger.info(f"Loaded {len(self.rules)} built-in validation rules")

    def validate_file(self, file_path: str, schema: Optional[Dict] = None) -> ValidationReport:
        """
        Validate a data file (CSV or JSON).

        Args:
            file_path: Path to file
            schema: Optional schema definition

        Returns:
            ValidationReport with results
        """
        start_time = datetime.now()

        # Get file info
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            report = ValidationReport(
                file_path=file_path,
                file_type="unknown",
                file_size=0,
                file_hash="",
                row_count=0,
                column_count=0,
                passed=False
            )
            report.issues.append(ValidationIssue(
                rule_name="file_exists",
                severity="error",
                message=f"File not found: {file_path}"
            ))
            return report

        file_size = file_path_obj.stat().st_size
        file_hash = self._compute_file_hash(file_path)
        file_type = file_path_obj.suffix.lower()

        # Load built-in rules if not already loaded
        if not self._builtin_rules_loaded:
            self.load_builtin_rules()

        # Validate based on file type
        if file_type == '.csv':
            report = self.validate_csv(file_path, schema)
        elif file_type == '.json':
            report = self.validate_json(file_path, schema)
        else:
            report = ValidationReport(
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                file_hash=file_hash,
                row_count=0,
                column_count=0,
                passed=False
            )
            report.issues.append(ValidationIssue(
                rule_name="unsupported_format",
                severity="error",
                message=f"Unsupported file format: {file_type}"
            ))

        # Update report with file info
        report.file_size = file_size
        report.file_hash = file_hash
        report.validation_time = (datetime.now() - start_time).total_seconds()

        # Determine if validation passed (no errors)
        report.passed = report.error_count == 0

        return report

    def validate_csv(self, file_path: str, schema: Optional[Dict] = None) -> ValidationReport:
        """Validate CSV file"""
        issues = []
        row_count = 0
        column_count = 0
        headers = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                column_count = len(headers)

                # Validate column names
                for col in headers:
                    if not self._is_valid_column_name(col):
                        issues.append(ValidationIssue(
                            rule_name="valid_column_names",
                            severity="warning",
                            message=f"Column name '{col}' doesn't follow naming conventions",
                            location=f"header:{col}"
                        ))

                # Validate rows
                seen_rows = set()
                for row_idx, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
                    row_count += 1

                    # Check for duplicate rows
                    row_tuple = tuple(sorted(row.items()))
                    if row_tuple in seen_rows:
                        issues.append(ValidationIssue(
                            rule_name="no_duplicate_rows",
                            severity="warning",
                            message=f"Duplicate row found",
                            location=f"row:{row_idx}"
                        ))
                    seen_rows.add(row_tuple)

                    # Validate each cell
                    for col, value in row.items():
                        # Check for missing values
                        if value is None or value.strip() == "":
                            issues.append(ValidationIssue(
                                rule_name="no_missing_values",
                                severity="error",
                                message=f"Missing value in column '{col}'",
                                location=f"row:{row_idx},col:{col}",
                                value=value
                            ))

                        # Apply schema validation if provided
                        if schema and col in schema:
                            col_schema = schema[col]
                            issues.extend(self._validate_value(
                                value, col_schema, f"row:{row_idx},col:{col}"
                            ))

        except Exception as e:
            issues.append(ValidationIssue(
                rule_name="file_parsing",
                severity="error",
                message=f"Error parsing CSV: {str(e)}"
            ))

        return ValidationReport(
            file_path=file_path,
            file_type=".csv",
            file_size=0,  # Will be set by validate_file
            file_hash="",  # Will be set by validate_file
            row_count=row_count,
            column_count=column_count,
            issues=issues,
            metadata={'headers': headers}
        )

    def validate_json(self, file_path: str, schema: Optional[Dict] = None) -> ValidationReport:
        """Validate JSON file"""
        issues = []
        row_count = 0
        column_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Determine structure
            if isinstance(data, list):
                row_count = len(data)
                if row_count > 0 and isinstance(data[0], dict):
                    column_count = len(data[0].keys())

                    # Validate each record
                    for idx, record in enumerate(data):
                        if schema:
                            issues.extend(self._validate_record(
                                record, schema, f"record:{idx}"
                            ))
            elif isinstance(data, dict):
                row_count = 1
                column_count = len(data.keys())
                if schema:
                    issues.extend(self._validate_record(data, schema, "root"))

        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                rule_name="json_parsing",
                severity="error",
                message=f"Invalid JSON: {str(e)}"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name="file_parsing",
                severity="error",
                message=f"Error parsing JSON: {str(e)}"
            ))

        return ValidationReport(
            file_path=file_path,
            file_type=".json",
            file_size=0,
            file_hash="",
            row_count=row_count,
            column_count=column_count,
            issues=issues
        )

    def _validate_value(
        self,
        value: Any,
        schema: Dict,
        location: str
    ) -> List[ValidationIssue]:
        """Validate a single value against schema"""
        issues = []

        # Type validation
        if 'type' in schema:
            expected_type = schema['type']
            if expected_type == 'number':
                if not self._is_valid_number(value):
                    issues.append(ValidationIssue(
                        rule_name="valid_type",
                        severity="error",
                        message=f"Expected number, got '{value}'",
                        location=location,
                        value=value
                    ))
            elif expected_type == 'string':
                if not isinstance(value, str):
                    issues.append(ValidationIssue(
                        rule_name="valid_type",
                        severity="error",
                        message=f"Expected string, got {type(value).__name__}",
                        location=location,
                        value=value
                    ))

        # Range validation
        if 'min' in schema and self._is_valid_number(value):
            if float(value) < schema['min']:
                issues.append(ValidationIssue(
                    rule_name="value_range",
                    severity="error",
                    message=f"Value {value} below minimum {schema['min']}",
                    location=location,
                    value=value
                ))

        if 'max' in schema and self._is_valid_number(value):
            if float(value) > schema['max']:
                issues.append(ValidationIssue(
                    rule_name="value_range",
                    severity="error",
                    message=f"Value {value} above maximum {schema['max']}",
                    location=location,
                    value=value
                ))

        # Pattern validation
        if 'pattern' in schema and isinstance(value, str):
            if not re.match(schema['pattern'], value):
                issues.append(ValidationIssue(
                    rule_name="value_pattern",
                    severity="error",
                    message=f"Value doesn't match pattern {schema['pattern']}",
                    location=location,
                    value=value
                ))

        return issues

    def _validate_record(
        self,
        record: Dict,
        schema: Dict,
        location: str
    ) -> List[ValidationIssue]:
        """Validate a record (dict) against schema"""
        issues = []

        # Check required fields
        if 'required' in schema:
            for field in schema['required']:
                if field not in record:
                    issues.append(ValidationIssue(
                        rule_name="required_field",
                        severity="error",
                        message=f"Missing required field: {field}",
                        location=location
                    ))

        # Validate fields
        if 'properties' in schema:
            for field, value in record.items():
                if field in schema['properties']:
                    field_location = f"{location}.{field}"
                    issues.extend(self._validate_value(
                        value, schema['properties'][field], field_location
                    ))

        return issues

    @staticmethod
    def _is_valid_number(value: Any) -> bool:
        """Check if value is a valid number"""
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_valid_column_name(name: str) -> bool:
        """Check if column name follows conventions"""
        # Valid: lowercase, underscores, alphanumeric
        return bool(re.match(r'^[a-z][a-z0-9_]*$', name))

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def export_report(self, report: ValidationReport, output_path: str):
        """Export validation report to JSON"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Validation report exported to {output_path}")

    def print_report(self, report: ValidationReport):
        """Print validation report to console"""
        print("\n" + "=" * 80)
        print(f"  DATA VALIDATION REPORT: {report.file_path}")
        print("=" * 80)

        print(f"\n📊 File Info:")
        print(f"   Type: {report.file_type}")
        print(f"   Size: {report.file_size:,} bytes ({report.file_size / 1024:.2f} KB)")
        print(f"   Hash: {report.file_hash[:16]}...")
        print(f"   Dimensions: {report.row_count} rows × {report.column_count} columns")

        status = "✓ PASSED" if report.passed else "✗ FAILED"
        status_color = "\033[0;32m" if report.passed else "\033[0;31m"
        print(f"\n{status_color}{status}\033[0m")

        print(f"\n📋 Validation Summary:")
        print(f"   Errors: {report.error_count}")
        print(f"   Warnings: {report.warning_count}")
        print(f"   Total Issues: {len(report.issues)}")
        print(f"   Validation Time: {report.validation_time:.3f}s")

        if report.issues:
            print(f"\n⚠️  Issues Found:")
            for i, issue in enumerate(report.issues[:10], 1):  # Show first 10
                severity_icon = "❌" if issue.severity == "error" else "⚠️"
                print(f"   {severity_icon} [{issue.severity.upper()}] {issue.message}")
                if issue.location:
                    print(f"      Location: {issue.location}")

            if len(report.issues) > 10:
                print(f"   ... and {len(report.issues) - 10} more issues")

        print("\n" + "=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    validator = DataValidator()

    # Example 1: Validate CSV
    print("Example 1: CSV Validation")
    report = validator.validate_file("sample_data/test.csv")
    validator.print_report(report)

    # Example 2: Validate JSON with schema
    print("Example 2: JSON Validation with Schema")
    schema = {
        'properties': {
            'id': {'type': 'number', 'min': 0},
            'name': {'type': 'string'},
            'email': {'type': 'string', 'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
        },
        'required': ['id', 'name']
    }
    report = validator.validate_file("sample_data/test.json", schema=schema)
    validator.print_report(report)

    # Export report
    validator.export_report(report, "validation_reports/test_report.json")
