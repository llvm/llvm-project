"""
This simple distributor performs code generation locally, creates the
"send-signal1" file, and then waits for the "send-signal2" file to appear
before exiting. It is intended to be used in tandem with test_temps.py.
Please see test_temps.py for more information.
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
