# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import os
import sys

import lldb


class TestFail(Exception):
    pass


def fail(msg: str):
    raise TestFail(msg)


def __lldb_init_module(debugger, internal_dict):
    debugger.SetAsync(False)

    try:
        target = debugger.GetSelectedTarget()

        if not target.IsValid():
            fail("Debugger target is not valid")

        process = target.LaunchSimple(None, None, os.getcwd())

        if not process.IsValid():
            fail("Failed to launch process")

        if process.GetExitStatus() != 0:
            fail("std::is_debugger_present() should be true and ret code == 0")
    except TestFail as e:
        print(f"{e}", file=sys.stderr)
        debugger.HandleCommand("quit 1")

    debugger.HandleCommand("quit 0")
