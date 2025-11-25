"""
ZEROPAIN Modules - Pharmaceutical Research Integration

This package contains molecular modeling, docking, ADMET prediction,
and patient simulation capabilities from the ZEROPAIN platform.

TEMPEST Compliance: All modules operate within TEMPEST security framework
"""

__version__ = "3.0.0"
__author__ = "ZEROPAIN Therapeutics / LAT5150DRVMIL Integration"

# Lazy imports for performance
_molecular_modules_loaded = False
_simulation_modules_loaded = False


def load_molecular_modules():
    """Lazy load molecular analysis modules"""
    global _molecular_modules_loaded
    if not _molecular_modules_loaded:
        from . import molecular
        _molecular_modules_loaded = True
    return True


def load_simulation_modules():
    """Lazy load patient simulation modules"""
    global _simulation_modules_loaded
    if not _simulation_modules_loaded:
        from . import simulation
        _simulation_modules_loaded = True
    return True
