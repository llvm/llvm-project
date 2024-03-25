#!/usr/bin/env python

import os, signal, sys, subprocess, tempfile
from android_common import *

ANDROID_TMPDIR = "/data/local/tmp/Output"

device_binary = host_to_device_path(sys.argv[0])


def build_env():
    args = []
    # Android linker ignores RPATH. Set LD_LIBRARY_PATH to Output dir.
    args.append("LD_LIBRARY_PATH=%s" % (ANDROID_TMPDIR,))
    for (key, value) in list(os.environ.items()):
        if key in ["ASAN_ACTIVATION_OPTIONS", "SCUDO_OPTIONS"] or key.endswith(
            "SAN_OPTIONS"
        ):
            args.append('%s="%s"' % (key, value.replace('"', '\\"')))
    return " ".join(args)


is_64bit = (
    str(subprocess.check_output(["file", sys.argv[0] + ".real"])).find("64-bit") != -1
)

device_env = build_env()
device_args = " ".join(sys.argv[1:])  # FIXME: escape?
device_stdout = device_binary + ".stdout"
device_stderr = device_binary + ".stderr"
device_exitcode = device_binary + ".exitcode"
ret = adb(
    [
        "shell",
        "cd %s && %s %s %s >%s 2>%s ; echo $? >%s"
        % (
            ANDROID_TMPDIR,
            device_env,
            device_binary,
            device_args,
            device_stdout,
            device_stderr,
            device_exitcode,
        ),
    ]
)
if ret != 0:
    sys.exit(ret)

sys.stdout.write(pull_from_device(device_stdout))
sys.stderr.write(pull_from_device(device_stderr))
retcode = int(pull_from_device(device_exitcode))
# If the device process died with a signal, do abort().
# Not exactly the same, but good enough to fool "not --crash".
if retcode > 128:
    os.kill(os.getpid(), signal.SIGABRT)
sys.exit(retcode)
