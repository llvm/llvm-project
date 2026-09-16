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

# When messages are attached together, the source locations to which they
# refer are not necessarily monotonically increasing. For example
#   error: foo.f90:10: There is a problem here         # line 10
#   because: foo.f90:12: This thing is invalid         # line 12 (attached)
#   error: foo.f90:11: There is another problem here   # line 11
# There is no way to represent that in the source file via ERROR annotations,
# so before running unified_diff "canonicalize" the list of messages into an
# order that corresponds to the line numbers.
#
# This also eliminates the issue with multiple messages emitted for the same
# line: they can now be "expected" in the test file in any order, e.g.
#   !ERROR: Not enough arguments in a call to foo
#   !ERROR: `foo` is a subroutine, not a function
#   a = foo()
# has the same effect as:
#   !ERROR: `foo` is a subroutine, not a function
#   !ERROR: Not enough arguments in a call to foo
#   a = foo()


def join_per_line_map(m):
    """Take a map {"line_no:": [message1, message2, ...], ...} and convert
    it into a newline-separated string that follows the line ordering.
    """
    # Sort messages for each line, and prepend the line number to each
    # message. Use numeric values of line numbers as keys to allow them
    # to be sorted numerically.
    sorted_lines_map = {
        int(k.rstrip(":")): [k + s for s in sorted(m[k])] for k in m.keys()
    }

    joined_lines_list = []
    for line in sorted(sorted_lines_map.keys()):
        joined_lines_list.append("\n".join(sorted_lines_map[line]))
    return "\n".join(joined_lines_list)


cm.check_args(sys.argv)
srcdir = cm.set_source(sys.argv[1])
with open(srcdir, "r", encoding="utf-8") as f:
    src = f.readlines()
diffs = ""
log = ""

flang_fc1 = cm.set_executable(sys.argv[2])
flang_fc1_args = sys.argv[3:]
flang_fc1_options = "-fsyntax-only"

# Compiles, and reads in the output from the compilation process
cmd = [flang_fc1, *flang_fc1_args, flang_fc1_options, str(srcdir)]
with tempfile.TemporaryDirectory() as tmpdir:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True,
            cwd=tmpdir,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as e:
        log = e.stderr
        if e.returncode >= 128:
            print(f"{log}")
            sys.exit(1)

# Cleans up the output from the compilation process to be easier to process
actual_per_line = dict()
for line in log.split("\n"):
    m = re.search(r"[^:]*:(\d+:).*(?:error|warning|portability|because):(.*)", line)
    if m:
        if re.search(r"warning: .*fold.*host", line):
            continue  # ignore host-dependent folding warnings
        line_colon = m.expand(r"\1")
        actual_per_line[line_colon] = actual_per_line.get(line_colon, []) + [
            m.expand(r"\2")
        ]

# Gets the expected errors and their line numbers
expect_per_line = dict()
errors = []
for i, line in enumerate(src, 1):
    m = re.search(r"(?:^\s*!\s*(?:ERROR|WARNING|PORTABILITY|BECAUSE): )(.*)", line)
    if m:
        errors.append(m.group(1))
        continue
    if errors:
        expect_per_line[f"{i}:"] = [f" {x}" for x in errors]
        errors = []

actual = join_per_line_map(actual_per_line)
expect = join_per_line_map(expect_per_line)

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
