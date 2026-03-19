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

# Detect whether LLDB is on the system and has Python scripting support.
# If so add a substitution to access it.

def check_lldb(cfg):
    lldb_path = shutil.which("lldb")
    if lldb_path is None:
        return False

    try:
        stdout = subprocess.check_output(
            [lldb_path, "--batch", "-o", "script -l python -- print(\"Has\", \"Python\", \"!\")"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError:
        return False

    # Check we actually ran the Python
    return "Has Python !" in stdout


features = [
    Feature(
        name="host-has-lldb-with-python",
        when=check_lldb,
        actions=[AddSubstitution("%{lldb}", lambda cfg: shutil.which("lldb"))],
    )
]
