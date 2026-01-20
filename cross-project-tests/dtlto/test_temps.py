"""Run command, wait for "send-signal1" file to exist, perform some action on
the DTLTO files (such as locking them) the action performed is based on the test
directory name, then create "send-signal2" file."""

import ctypes, os, subprocess, sys, time
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
p = subprocess.Popen(sys.argv[2:])

while not os.path.exists("send-signal1") and p.poll() is None:
    time.sleep(0.05)
if p.poll() is not None:
    sys.exit(1)

if output_dir == "locked":
    # lock any files in the output directory.
    for f in Path(output_dir).iterdir():
        if f.is_file():
            lock_no_delete_share(str(f))

if output_dir == "removed":
    # remove non-essential files in the output directory.
    for f in Path(output_dir).iterdir():
        if f.is_file() and not f.name.endswith("native.o"):
            f.unlink()

Path("send-signal2").touch()

sys.exit(p.wait())
