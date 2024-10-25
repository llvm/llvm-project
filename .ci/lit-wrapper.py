#! /usr/bin/env/python3

import sys
import os
import subprocess
import tempfile

result_path = None
for idx, option in enumerate(sys.argv):
    if option == "--xunit-xml-output":
        result_path = sys.argv[idx + 1]
        break

dirname, _ = os.path.split(os.path.abspath(__file__))
res = subprocess.run(
    [sys.executable, os.path.join(dirname, "llvm-lit-actual.py"), *sys.argv[1:]],
    check=False,
)

if result_path is not None:
    with open(result_path, "rb") as results_file:
        filename, ext = os.path.splitext(os.path.basename(result_path))
        fd, _ = tempfile.mkstemp(
            suffix=ext, prefix=f"{filename}.", dir=os.path.dirname(result_path)
        )
        with os.fdopen(fd, "wb") as out:
            out.write(results_file.read())

sys.exit(res.returncode)
