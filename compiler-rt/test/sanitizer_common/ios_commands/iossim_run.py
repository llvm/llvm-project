#!/usr/bin/env python3

import glob, os, shlex, sys, subprocess


device_id = os.environ.get("SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER")
iossim_run_verbose = os.environ.get("SANITIZER_IOSSIM_RUN_VERBOSE")
wait_for_debug = os.environ.get("SANITIZER_IOSSIM_RUN_WAIT_FOR_DEBUGGER")

if not device_id:
    raise EnvironmentError(
        "Specify SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER to select which simulator to use."
    )

for e in [
    "ASAN_OPTIONS",
    "TSAN_OPTIONS",
    "UBSAN_OPTIONS",
    "LSAN_OPTIONS",
    "APPLE_ASAN_INIT_FOR_DLOPEN",
    "ASAN_ACTIVATION_OPTIONS",
    "MallocNanoZone",
]:
    simctl_version = "SIMCTL_CHILD_" + e
    # iossim_env.py might have already set these using arguments it was given
    # (and that we can't see from inside this script). Don't overwrite them!
    if e in os.environ and simctl_version not in os.environ:
        os.environ[simctl_version] = os.environ[e]

find_atos_cmd = "xcrun -sdk iphonesimulator -f atos"
atos_path = (
    subprocess.run(
        find_atos_cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    .stdout.decode()
    .strip()
)
for san in ["ASAN", "TSAN", "UBSAN", "LSAN"]:
    os.environ[f"SIMCTL_CHILD_{san}_SYMBOLIZER_PATH"] = atos_path

prog = sys.argv[1]
exit_code = None
if prog == "rm":
    # The simulator and host actually share the same file system so we can just
    # execute directly on the host.
    rm_args = []
    for arg in sys.argv[2:]:
        if "*" in arg or "?" in arg:
            # Don't quote glob pattern
            rm_args.append(arg)
        else:
            rm_args.append(shlex.quote(arg))
    rm_cmd_line = ["/bin/rm"] + rm_args
    rm_cmd_line_str = " ".join(rm_cmd_line)
    # We use `shell=True` so that any wildcard globs get expanded by the shell.

    if iossim_run_verbose:
        print("RUNNING: \t{}".format(rm_cmd_line_str), flush=True)

    exitcode = subprocess.call(rm_cmd_line_str, shell=True)

else:
    cmd = ["xcrun", "simctl", "spawn", "--standalone"]

    if wait_for_debug:
        cmd.append("--wait-for-debugger")

    cmd.append(device_id)
    cmd += sys.argv[1:]

    if iossim_run_verbose:
        print("RUNNING: \t{}".format(" ".join(cmd)), flush=True)

    exitcode = subprocess.call(cmd)
if exitcode > 125:
    exitcode = 126
sys.exit(exitcode)
