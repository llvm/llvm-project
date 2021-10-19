#!/usr/bin/env python3

"""Compiles a source file and compares generated .mod files against expected.

Parameters:
    sys.argv[1]: a source file with contains the input and expected output
    sys.argv[2]: the Flang frontend driver
    sys.argv[3:]: Optional arguments to the Flang frontend driver"""

import sys
import re
import os
import tempfile
import subprocess
import glob
import common as cm

from pathlib import Path
from difflib import unified_diff

cm.check_args_long(sys.argv)
srcdir = Path(sys.argv[1])
sources = list(glob.iglob(str(srcdir)))
sources = sorted(sources)
diffs = ""

flang_fc1 = cm.set_executable(sys.argv[2])
flang_fc_args = sys.argv[3:]
flang_fc1_options = "-fsyntax-only"

with tempfile.TemporaryDirectory() as tmpdir:
    for src in sources:
        src = Path(src).resolve()
        actual = ""
        expect = ""
        expected_files = set()
        actual_files = set()

        if not src.is_file():
            cm.die(src)

        prev_files = set(os.listdir(tmpdir))
        cmd = [flang_fc1, *flang_fc_args, flang_fc1_options, str(src)]
        proc = subprocess.check_output(cmd, stderr=subprocess.PIPE,
                                       universal_newlines=True, cwd=tmpdir)
        actual_files = set(os.listdir(tmpdir)).difference(prev_files)

        # The first 3 bytes of the files are an UTF-8 BOM
        with open(src, 'r', encoding="utf-8", errors="strict") as f:
            for line in f:
                m = re.search(r"^!Expect: (.*)", line)
                if m:
                    expected_files.add(m.group(1))

        extra_files = actual_files.difference(expected_files)
        if extra_files:
            print(f"Unexpected .mod files produced: {extra_files}")
            sys.exit(1)

        for mod in expected_files:
            mod = Path(tmpdir).joinpath(mod)
            if not mod.is_file():
                print(f"Compilation did not produce expected mod file: {mod}")
                sys.exit(1)
            with open(mod, 'r', encoding="utf-8", errors="strict") as f:
                for line in f:
                    if "!mod$" in line:
                        continue
                    actual += line

            with open(src, 'r', encoding="utf-8", errors="strict") as f:
                for line in f:
                    if f"!Expect: {mod.name}" in line:
                        for line in f:
                            if re.match(r"^$", line):
                                break
                            m = re.sub(r"^!", "", line.lstrip())
                            expect += m

            diffs = "\n".join(unified_diff(actual.replace(" ", "").split("\n"),
                                           expect.replace(" ", "").split("\n"),
                                           fromfile=mod.name, tofile="Expect", n=999999))

            if diffs != "":
                print(diffs)
                print()
                print("FAIL")
                sys.exit(1)

print()
print("PASS")

