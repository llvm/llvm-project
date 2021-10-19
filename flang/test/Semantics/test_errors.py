#!/usr/bin/env python3

"""Compiles a source file and checks errors against those listed in the file.

Parameters:
    sys.argv[1]: a source file with contains the input and expected output
    sys.argv[2]: the Flang frontend driver
    sys.argv[3:]: Optional arguments to the Flang frontend driver"""

import sys
import re
import tempfile
import subprocess
import common as cm

from difflib import unified_diff

cm.check_args(sys.argv)
srcdir = cm.set_source(sys.argv[1])
with open(srcdir, 'r') as f:
    src = f.readlines()
actual = ""
expect = ""
diffs = ""
log = ""

flang_fc1 = cm.set_executable(sys.argv[2])
flang_fc1_args = sys.argv[3:]
flang_fc1_options = "-fsyntax-only"

# Compiles, and reads in the output from the compilation process
cmd = [flang_fc1, *flang_fc1_args, flang_fc1_options, str(srcdir)]
with tempfile.TemporaryDirectory() as tmpdir:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, universal_newlines=True, cwd=tmpdir)
    except subprocess.CalledProcessError as e:
        log = e.stderr
        if e.returncode >= 128:
            print(f"{log}")
            sys.exit(1)

# Cleans up the output from the compilation process to be easier to process
for line in log.split('\n'):
    m = re.search(r"[^:]*:(\d+:).*(?:error:)(.*)", line)
    if m:
        actual += m.expand(r"\1\2\n")

# Gets the expected errors and their line number
errors = []
for i, line in enumerate(src, 1):
    m = re.search(r"(?:^\s*!ERROR: )(.*)", line)
    if m:
        errors.append(m.group(1))
        continue
    if errors:
        for x in errors:
            expect += f"{i}: {x}\n"
        errors = []

# Compares the expected errors with the compiler errors
for line in unified_diff(actual.split("\n"), expect.split("\n"), n=0):
    line = re.sub(r"(^\-)(\d+:)", r"\nactual at \g<2>", line)
    line = re.sub(r"(^\+)(\d+:)", r"\nexpect at \g<2>", line)
    diffs += line

if diffs != "":
    print(diffs)
    print()
    print("FAIL")
    sys.exit(1)
else:
    print()
    print("PASS")

