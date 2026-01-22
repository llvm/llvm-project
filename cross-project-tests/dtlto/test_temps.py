"""
Run a command, wait for the "send-signal1" file to appear, then perform an
action on the DTLTO files in the output directory (for example, locking
them). The output directory is specified by sys.argv[1], and the action to
perform is specified by sys.argv[2]. Finally, create the "send-signal2" file.

This script works in tandem with local_codegen_and_wait.py. By coordinating
via the "send-signal*" files, the scripts ensure that the action is performed
after all DTLTO backend compilations have completed but before DTLTO itself
finishes. At this stage, DTLTO temporary files have not yet been cleaned up.
"""

import ctypes, os, subprocess, sys, time, signal
from ctypes import wintypes
from pathlib import Path

if os.name == "nt":
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


output_dir = sys.argv[1]
action = sys.argv[2]

kwargs = {}
if action == "kill":
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP is used so that p.send_signal(CTRL_BREAK_EVENT)
        # does not get sent to the LIT processes that are running the test.
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        # Makes the child a process-group leader so os.killpg(p.pid, SIGINT) works.
        kwargs = {"start_new_session": True}

p = subprocess.Popen(sys.argv[3:], **kwargs)

while not os.path.exists("send-signal1") and p.poll() is None:
    time.sleep(0.05)
if p.poll() is not None:
    sys.exit(1)

if action == "kill":
    if os.name == "nt":
        # Note that CTRL_C_EVENT does not appear to work for clang.
        p.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(p.pid, signal.SIGINT)

if action == "lock":
    print("Lock any files in the output directory.")
    for f in Path(output_dir).iterdir():
        if f.is_file():
            lock_no_delete_share(str(f))

if action == "remove":
    print("Remove non-essential files in the output directory.")
    for f in Path(output_dir).iterdir():
        if f.is_file() and not f.name.endswith("native.o"):
            f.unlink()

Path("send-signal2").touch()

if action == "kill":
    p.wait()
    sys.exit(0 if p.returncode != 0 else 1)
else:
    sys.exit(p.wait())
