#!/usr/bin/env python3

"""Compiles a source file with "-fdebug-unparse-with-symbols' and verifies
we get the right symbols in the output, i.e. the output should be
the same as the input, except for the copyright comment.

Parameters:
    sys.argv[1]: a source file with contains the input and expected output
    sys.argv[2]: the Flang frontend driver
    sys.argv[3:]: Optional arguments to the Flang frontend driver"""

import sys
import tempfile
import re
import subprocess
import common as cm

from difflib import unified_diff

cm.check_args(sys.argv)
src = cm.set_source(sys.argv[1])
diff1 = ""
diff2 = ""

flang_fc1 = cm.set_executable(sys.argv[2])
flang_fc1_args = sys.argv[3:]
flang_fc1_options = "-fdebug-unparse-with-symbols"

# Strips out blank lines and all comments except for "!DEF:", "!REF:", "!$acc" and "!$omp"
with open(src, "r") as text_in:
    for line in text_in:
        text = re.sub(r"!(?![DR]EF:|\$omp|\$acc).*", "", line)
        text = re.sub(r"^\s*$", "", text)
        diff1 += text

# Strips out "!DEF:" and "!REF:" comments
for line in diff1:
    text = re.sub(r"![DR]EF:.*", "", line)
    diff2 += text

# Compiles, inserting comments for symbols:
cmd = [flang_fc1, *flang_fc1_args, flang_fc1_options]
with tempfile.TemporaryDirectory() as tmpdir:
    diff3 = subprocess.check_output(
        cmd, input=diff2, universal_newlines=True, cwd=tmpdir
    )

# Removes all whitespace to compare differences in files
diff1 = diff1.replace(" ", "")
diff3 = diff3.replace(" ", "")
diff_check = ""

# Compares the input with the output
diff_check = "\n".join(
    unified_diff(
        diff1.split("\n"),
        diff3.split("\n"),
        n=999999,
        fromfile="Expected_output",
        tofile="Actual_output",
    )
)

if diff_check != "":
    print(diff_check.replace(" ", ""))
    print()
    print("FAIL")
    sys.exit(1)
else:
    print()
    print("PASS")
