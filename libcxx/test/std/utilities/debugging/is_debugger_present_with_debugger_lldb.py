# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import lldb

test_failures = 0

# Sometimes the inital run command can fail to trace the process.
# (e.g. you don't have ptrace permissions)
# In these cases gdb still sends us an exited event so we cannot
# see what "run" printed to check for a warning message, since
# we get taken to our exit handler before we can look.
# Instead check that at least one test has been run by the time
# we exit.
has_run_tests = False


def breakpoint_handler(frame, bp_loc, internal_dict):
    global has_run_tests

    try:
        has_run_tests = True

        module = frame.GetModule()
        filename = module.compile_units[0].file
        line = frame.GetLineEntry().GetLine()
        parent = frame.get_parent_frame()
        expectation_val = parent.FindVariable("isDebuggerPresent")

        if expectation_val is None or expectation_val.value == "false":
            global test_failures

            print(f"FAIL: {filename}:{line}")
            print("`isDebuggerPresent` value is `false`, value should be `true`")

            test_failures += 1
        else:
            print(f"PASS: {filename}:{line}")

    except RuntimeError as e:
        # At this point, lots of different things could be wrong, so don't try to
        # recover or figure it out. Don't exit either, because then it's
        # impossible to debug the framework itself.

        print("FAIL: Something is wrong in the test framework.")
        print(str(e))

        test_failures += 1


def exit_handler(debugger):
    """Exit handler"""

    global test_failures
    global has_run_tests

    if not has_run_tests:
        print("FAILED test program did not run correctly, check lldb warnings")
        test_failures = -1
    elif test_failures:
        test_failures -= 1
        print(f"FAILED {test_failures} cases")

    debugger.HandleCommand(f"exit {test_failures}")


def __lldb_init_module(debugger, internal_dict):
    global test_failures

    target = debugger.GetSelectedTarget()
    test_bp = target.BreakpointCreateByName("StopForDebugger")
    test_bp.SetScriptCallbackFunction(
        "is_debugger_present_with_debugger_lldb.breakpoint_handler"
    )
    test_bp.enabled = True

    debugger.HandleCommand("run")

    test_failures += 1

    exit_handler(debugger)
