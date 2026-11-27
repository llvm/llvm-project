"""
This script works in tandem with local_codegen_and_wait.py. By coordinating
via the "send-signal*" files, the scripts ensure that the requested action is
performed after all DTLTO backend compilations have completed but before
DTLTO itself finishes. At this point, DTLTO temporary files have been
created but have not yet been cleaned up.

Usage:
  %python test_temps.py <output_dir> <action> <command...>

Run <command>, which must be a ThinLTO link invocation that uses
local_codegen_and_wait.py as the ThinLTO distributor. The script waits for
the "send-signal1" file to appear, performs <action>, and then creates the
"send-signal2" file to allow the link to continue.

Actions:
  kill    Send an interrupt to cause <command> to terminate after the
          ThinLTO backend compilations complete.
  lock    (Windows only) Hold open handles to DTLTO temporary files in
          <output_dir> to prevent their deletion after the ThinLTO backend
          compilations complete. This action is not supported on Linux, as
          there is no reliable mechanism to prevent file deletion that is
          guaranteed to be released when the script exits (AFAICT).
  remove  Delete non-essential files in <output_dir> after the ThinLTO
          backend compilations complete.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

IS_WIN = os.name == "nt"
SIGNAL1 = Path("send-signal1")
SIGNAL2 = Path("send-signal2")

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    CreateFileW = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    CreateFileW.argtypes = [
        wintypes.LPCWSTR,  # lpFileName
        wintypes.DWORD,  # dwDesiredAccess
        wintypes.DWORD,  # dwShareMode
        wintypes.LPVOID,  # lpSecurityAttributes
        wintypes.DWORD,  # dwCreationDisposition
        wintypes.DWORD,  # dwFlagsAndAttributes
        wintypes.HANDLE,  # hTemplateFile
    ]
    CreateFileW.restype = wintypes.HANDLE


def lock_no_delete_share(path):
    # Windows-specific: deny FILE_SHARE_DELETE by omitting it from dwShareMode.
    h = CreateFileW(
        path,
        0x80000000,  # GENERIC_READ
        0x00000003,  # FILE_SHARE_READ/WRITE (no FILE_SHARE_DELETE)
        None,  # lpSecurityAttributes
        3,  # OPEN_EXISTING
        0,  # dwFlagsAndAttributes
        None,  # hTemplateFile
    )
    if h == wintypes.HANDLE(-1).value:
        err = ctypes.get_last_error()
        raise OSError(err, f"CreateFileW failed ({err}) for: {path}")
    return h


output_dir = Path(sys.argv[1])
action = sys.argv[2]

# "lock" is Windows-only; fail early if invoked elsewhere.
if action == "lock" and not IS_WIN:
    print("error: action 'lock' is only supported on Windows", file=sys.stderr)
    sys.exit(1)

# Remove any pre-existing signal files from previous script runs.
SIGNAL1.unlink(missing_ok=True)
SIGNAL2.unlink(missing_ok=True)

kwargs = {}
if action == "kill":
    if IS_WIN:
        # CREATE_NEW_PROCESS_GROUP is used so that p.send_signal(CTRL_BREAK_EVENT)
        # does not get sent to the LIT processes that are running the test.
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        # Makes the child a process-group leader so os.killpg(p.pid, SIGINT) works.
        kwargs = {"start_new_session": True}

p = subprocess.Popen(sys.argv[3:], **kwargs)

while not SIGNAL1.exists() and p.poll() is None:
    time.sleep(0.05)
if p.poll() is not None:
    sys.exit(1)

if action == "kill":
    if IS_WIN:
        # Note that CTRL_C_EVENT does not appear to work for clang.
        p.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(p.pid, signal.SIGINT)

if action == "lock":
    print("Lock any files in the output directory.")
    for f in output_dir.iterdir():
        if f.is_file():
            lock_no_delete_share(str(f))

if action == "remove":
    print("Remove non-essential files in the output directory.")
    for f in output_dir.iterdir():
        if f.is_file() and not f.name.endswith("native.o"):
            f.unlink()

SIGNAL2.touch()

if action == "kill":
    # Expect termination: succeed if the child fails, fail if it exits cleanly.
    p.wait()
    sys.exit(0 if p.returncode != 0 else 1)

sys.exit(p.wait())
