# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import Feature, AddSubstitution
import shutil
import subprocess


features = []


# Detect whether dbx debugger (available on AIX and others) is on the system.
def check_dbx(cfg):
    dbx_path = shutil.which("dbx")
    if dbx_path is None:
        return False

    return True


features += [
    Feature(
        name="host-has-dbx",
        when=check_dbx,
        actions=[AddSubstitution("%{dbx}", lambda cfg: shutil.which("dbx"))],
    )
]


# Detect whether LLDB debugger is on the system.
def check_lldb(cfg):
    lldb_path = shutil.which("lldb")
    if lldb_path is None:
        return False

    return True


features += [
    Feature(
        name="host-has-lldb",
        when=check_lldb,
        actions=[AddSubstitution("%{lldb}", lambda cfg: shutil.which("lldb"))],
    )
]


# Detect whether GDB is on the system, has Python scripting and supports
# adding breakpoint commands. If so add a substitution to access it.
def check_gdb(cfg):
    gdb_path = shutil.which("gdb")
    if gdb_path is None:
        return False

    # Check that we can set breakpoint commands, which was added in 8.3.
    # Using the quit command here means that gdb itself exits, not just
    # the "python <...>" command.
    test_src = """\
try:
  gdb.Breakpoint(\"main\").commands=\"foo\"
except AttributeError:
  gdb.execute(\"quit 1\")
gdb.execute(\"quit\")"""

    try:
        stdout = subprocess.check_output(
            [gdb_path, "-ex", "python " + test_src, "--batch"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError:
        # We can't set breakpoint commands
        return False

    # Check we actually ran the Python
    return not "Python scripting is not supported" in stdout


features += [
    Feature(
        name="host-has-gdb-with-python",
        when=check_gdb,
        actions=[AddSubstitution("%{gdb}", lambda cfg: shutil.which("gdb"))],
    )
]
