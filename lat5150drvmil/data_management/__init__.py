"""
Data Management Package
-----------------------
Comprehensive data management, validation, and governance for LAT5150DRVMIL.

Modules:
- validators.data_validator: Automated data validation framework
- policies: Data governance policies and procedures

Usage:
    from data_management.validators.data_validator import DataValidator

    validator = DataValidator()
    report = validator.validate_file("sample_data/test.csv")
    validator.print_report(report)
"""

from .validators.data_validator import (
    DataValidator,
    ValidationRule,
    ValidationIssue,
    ValidationReport
)

__all__ = [
    'DataValidator',
    'ValidationRule',
    'ValidationIssue',
    'ValidationReport'
]

__version__ = '1.0.0'
