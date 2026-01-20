"""Perform codegen locally, create "send-signal1" file and wait
for "send-signal2" file to exist before exiting."""

import json, subprocess, sys, time, os, pathlib

# Load the DTLTO information from the input JSON file.
data = json.loads(pathlib.Path(sys.argv[-1]).read_bytes())

# Iterate over the jobs and execute the codegen tool.
for job in data["jobs"]:
    subprocess.check_call(data["common"]["args"] + job["args"])

pathlib.Path("send-signal1").touch()

while not os.path.exists("send-signal2"):
    time.sleep(0.05)
