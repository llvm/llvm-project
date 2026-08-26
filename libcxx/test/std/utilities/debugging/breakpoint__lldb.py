# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import os
import platform
import sys

import lldb


class TestFail(Exception):
    pass


def fail(msg: str):
    raise TestFail(f"{msg}")


def run_test(debugger):
    debugger.SetAsync(False)

    target = debugger.GetSelectedTarget()

    if not target.IsValid():
        fail("Invalid LLDB target")

    process = target.LaunchSimple(None, None, os.getcwd())

    if not process.IsValid():
        fail("Failed to launch process")

    platform_name = platform.system()

    stop_reason = (
        lldb.eStopReasonException
        if platform_name in ("Darwin", "Windows")
        else lldb.eStopReasonSignal
    )

    stopped_thread = None
    for t in process.threads:
        if t.GetStopReason() == stop_reason:
            stopped_thread = t
            break

    if stopped_thread is None:
        fail("Could not find thread stopped by std::breakpoint")

    is_wow64_breakpoint = (  # 32 bit Windows work around
        platform_name == "Windows"
        and "0x4000001f" in stopped_thread.GetStopDescription(256)
    )

    if is_wow64_breakpoint:
        print("Stepping past WOW64 breakpoint", file=sys.stderr)
        process.Continue()
        for t in process.threads:
            if t.GetStopReason() == stop_reason:
                stopped_thread = t
                break

    found_main = False
    for frame in stopped_thread:
        if not frame.IsValid():
            fail(f"Frame is not valid")
        if frame.GetFunctionName() == "main":
            found_main = True
            break

    if not found_main:
        fail("Did not find main in stopped frames")

    error = process.Continue()
    if error.Fail():
        fail(f"Failed to continue: {error.GetCString()}")

    if process.GetExitStatus() != 0:
        fail(f"Unexpected exit status: {process.GetExitStatus()}")


def __lldb_init_module(debugger, internal_dict):
    exit_code = 0
    try:
        run_test(debugger)
    except Exception as e:
        print(f"Test Failure: {e}", file=sys.stderr)
        exit_code = 1

    debugger.HandleCommand(f"quit {exit_code}")
