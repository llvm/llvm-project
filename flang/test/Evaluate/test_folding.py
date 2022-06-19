#!/usr/bin/env python3

"""This script verifies expression folding.
It compiles a source file with '-fdebug-dump-symbols'
and looks for parameter declarations to check
they have been folded as expected.
To check folding of an expression EXPR,
the fortran program passed to this script
must contain the following:

  logical, parameter :: test_x = <compare EXPR to expected value>

This script will test that all parameter
with a name starting with "test_"
have been folded to .true.
For instance, acos folding can be tested with:

  real(4), parameter :: res_acos = acos(0.5_4)
  real(4), parameter :: exp_acos = 1.047
  logical, parameter :: test_acos = abs(res_acos - exp_acos).LE.(0.001_4)

There are two kinds of failure:
  - test_x is folded to .false..
    This means the expression was folded
    but the value is not as expected.
  - test_x is not folded (it is neither .true. nor .false.).
    This means the compiler could not fold the expression.

Parameters:
    sys.argv[1]: a source file with contains the input and expected output
    sys.argv[2]: the Flang frontend driver
    sys.argv[3:]: Optional arguments to the Flang frontend driver"""

import os
import sys
import tempfile
import re
import subprocess

from difflib import unified_diff
from pathlib import Path

def check_args(args):
    """Verifies that the number is arguments passed is correct."""
    if len(args) < 3:
        print(f"Usage: {args[0]} <fortran-source> <flang-command>")
        sys.exit(1)

def set_source(source):
    """Sets the path to the source files."""
    if not Path(source).is_file():
        print(f"File not found: {src}")
        sys.exit(1)
    return Path(source)

def set_executable(exe):
    """Sets the path to the Flang frontend driver."""
    if not Path(exe).is_file():
        print(f"Flang was not found: {exe}")
        sys.exit(1)
    return str(Path(exe))

check_args(sys.argv)
cwd = os.getcwd()
srcdir = set_source(sys.argv[1]).resolve()
with open(srcdir, 'r', encoding="utf-8") as f:
    src = f.readlines()
src1 = ""
src2 = ""
src3 = ""
src4 = ""
messages = ""
actual_warnings = ""
expected_warnings = ""
warning_diffs = ""

flang_fc1 = set_executable(sys.argv[2])
flang_fc1_args = sys.argv[3:]
flang_fc1_options = ""
LIBPGMATH = os.getenv('LIBPGMATH')
if LIBPGMATH:
    flang_fc1_options = ["-fdebug-dump-symbols", "-DTEST_LIBPGMATH"]
    print("Assuming libpgmath support")
else:
    flang_fc1_options = ["-fdebug-dump-symbols"]
    print("Not assuming libpgmath support")

cmd = [flang_fc1, *flang_fc1_args, *flang_fc1_options, str(srcdir)]
with tempfile.TemporaryDirectory() as tmpdir:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, universal_newlines=True, cwd=tmpdir)
    src1 = proc.stdout
    messages = proc.stderr

for line in src1.split("\n"):
    m = re.search(r"(\w*)(?=, PARAMETER).*init:(.*)", line)
    if m:
        src2 += f"{m.group(1)} {m.group(2)}\n"

for line in src2.split("\n"):
    m = re.match(r"test_*", line)
    if m:
        src3 += f"{m.string}\n"

for passed_results, line in enumerate(src3.split("\n")):
    m = re.search(r"\.false\._.$", line)
    if m:
        src4 += f"{line}\n"

for line in messages.split("\n"):
    m = re.search(r"[^:]*:(\d*):\d*: (.*)", line)
    if m:
        actual_warnings += f"{m.group(1)}: {m.group(2)}\n"

passed_warnings = 0
warnings = []
for i, line in enumerate(src, 1):
    m = re.search(r"(?:!WARN:)(.*)", line)
    if m:
        warnings.append(m.group(1))
        continue
    if warnings:
        for x in warnings:
            passed_warnings += 1
            expected_warnings += f"{i}:{x}\n"
        warnings = []

for line in unified_diff(actual_warnings.split("\n"),
                         expected_warnings.split("\n"), n=0):
    line = re.sub(r"(^\-)(\d+:)", r"\nactual at \g<2>", line)
    line = re.sub(r"(^\+)(\d+:)", r"\nexpect at \g<2>", line)
    warning_diffs += line

if src4 or warning_diffs:
    print("Folding test failed:")
    # Prints failed tests, including parameters with the same
    # suffix so that more information can be obtained by declaring
    # expected_x and result_x
    if src4:
        for line in src4.split("\n"):
            m = re.match(r"test_(\w+)", line)
            if m:
                for line in src2.split("\n"):
                    if m.group(1) in line:
                        print(line)
    if warning_diffs:
        print(warning_diffs)
    print()
    print("FAIL")
    sys.exit(1)
else:
    print()
    print(f"All {passed_results+passed_warnings} tests passed")
    print("PASS")

