import os, sys, subprocess, tempfile
import time

ANDROID_TMPDIR = "/data/local/tmp/Output"
ADB = os.environ.get("ADB", "adb")

verbose = False
if os.environ.get("ANDROID_RUN_VERBOSE") == "1":
    verbose = True


def host_to_device_path(path):
    rel = os.path.relpath(path, "/")
    dev = os.path.join(ANDROID_TMPDIR, rel)
    return dev


def adb(args, attempts=1, timeout_sec=600):
    if verbose:
        print(args)
    with tempfile.TemporaryFile(mode="w+") as out:
        ret = 255
        while attempts > 0 and ret != 0:
            attempts -= 1
            ret = subprocess.call(
                ["timeout", str(timeout_sec), ADB] + args,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        if ret != 0:
            print("adb command failed", args)
            out.seek(0)
            print(out.read())
    return ret


def pull_from_device(path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = os.path.join(tmp_dir, "pulled")
        adb(["pull", path, tmp], 5, 60)
        with open(tmp, "r") as f:
            return f.read()


def push_to_device(path):
    dst_path = host_to_device_path(path)
    adb(["push", path, dst_path], 5, 60)
