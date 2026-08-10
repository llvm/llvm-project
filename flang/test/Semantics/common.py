"""Provides common functionality to the test scripts."""

import os
import sys
from pathlib import Path


def set_source(source):
    """Checks whether the source file exists and returns its path."""
    if not Path(source).is_file():
        die(source)
    return Path(source)


def set_executable(executable):
    """Checks whether a Flang executable has been set and returns its path."""
    flang_fc1 = Path(executable)
    if not flang_fc1.is_file():
        die(flang_fc1)
    return str(flang_fc1)


def set_temp(tmp):
    """Sets a temporary directory or creates one if it doesn't exist."""
    os.makedirs(Path(tmp), exist_ok=True)
    return Path(tmp)


def die(file=None):
    """Used in other functions."""
    if file is None:
        print(f"{sys.argv[0]}: FAIL")
    else:
        print(f"{sys.argv[0]}: File not found: {file}")
    sys.exit(1)


def check_args(args):
    """Verifies that 2 arguments have been passed."""
    if len(args) < 3:
        print(f"Usage: {args[0]} <fortran-source> <flang-command>")
        sys.exit(1)


def check_args_long(args):
    """Verifies that 3 arguments have been passed."""
    if len(args) < 4:
        print(f"Usage: {args[0]} <fortran-source> <temp-test-dir> <flang-command>")
        sys.exit(1)
