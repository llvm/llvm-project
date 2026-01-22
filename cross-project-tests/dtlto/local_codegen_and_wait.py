"""
This simple distributor performs code generation locally, creates the
"send-signal1" file, and then waits for the "send-signal2" file to appear
before exiting. It is intended to be used in tandem with test_temps.py.

By coordinating via the "send-signal*" files, the scripts ensure that the
requested actions are performed after all DTLTO backend compilations have
completed but before DTLTO itself finishes. At this point, DTLTO temporary
files have not yet been cleaned up.
"""

import json, subprocess, sys, time, os, pathlib

# Load the DTLTO information from the input JSON file.
data = json.loads(pathlib.Path(sys.argv[-1]).read_bytes())

# Iterate over the jobs and execute the codegen tool.
for job in data["jobs"]:
    subprocess.check_call(data["common"]["args"] + job["args"])

pathlib.Path("send-signal1").touch()

while not os.path.exists("send-signal2"):
    time.sleep(0.05)
