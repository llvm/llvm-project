# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from __future__ import print_function
import gdb

# https://sourceware.org/gdb/current/onlinedocs/gdb.html/Python.html

test_failures = 0

# Sometimes the inital run command can fail to trace the process.
# (e.g. you don't have ptrace permissions)
# In these cases gdb still sends us an exited event so we cannot
# see what "run" printed to check for a warning message, since
# we get taken to our exit handler before we can look.
# Instead check that at least one test has been run by the time
# we exit.
has_run_tests = False


class CheckResult(gdb.Command):
    """GDB Tester"""

    def __init__(self):
        super(CheckResult, self).__init__("check_is_debugger_present", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        global has_run_tests

        try:
            has_run_tests = True

            # Stack frame is:
            # 0. StopForDebugger
            # 1. Check `isDebuggerPresent`

            compare_frame = gdb.newest_frame().older()
            testcase_frame = compare_frame.older()
            test_loc = testcase_frame.find_sal()

            # Ignore the convenience variable name and newline

            # value = value_str[value_str.find("= ") + 2 : -1]
            gdb.newest_frame().select()
            expectation_val = compare_frame.read_var("isDebuggerPresent")

            if not expectation_val:
                global test_failures

                print("FAIL: " + test_loc.symtab.filename + ":" + str(test_loc.line))
                print("`isDebuggerPresent` value is `false`, value should be `true`")

                test_failures += 1
            else:
                print("PASS: " + test_loc.symtab.filename + ":" + str(test_loc.line))

        except RuntimeError as e:
            # At this point, lots of different things could be wrong, so don't try to
            # recover or figure it out. Don't exit either, because then it's
            # impossible to debug the framework itself.

            print("FAIL: Something is wrong in the test framework.")
            print(str(e))

            test_failures += 1


def exit_handler(event=None):
    """Exit handler"""

    global test_failures
    global has_run_tests

    if not has_run_tests:
        print("FAILED test program did not run correctly, check gdb warnings")
        test_failures = -1
    elif test_failures:
        print(f"FAILED {test_failures} cases")
    exit(test_failures)

def main():
    # Start code executed at load time

    # Disable terminal paging

    gdb.execute("set height 0")
    gdb.execute("set python print-stack full")

    test = CheckResult()
    test_bp = gdb.Breakpoint("StopForDebugger")
    test_bp.enabled = True
    test_bp.silent = True
    test_bp.commands = """check_is_debugger_present
    continue"""

    # "run" won't return if the program exits; ensure the script regains control.

    gdb.events.exited.connect(exit_handler)
    gdb.execute("run")

    # If the program didn't exit, something went wrong, but we don't
    # know what. Fail on exit.

    test_failures += 1
    exit_handler(None)

    print(f"Test failures count: {test_failures}")

if __name__ == "__main__":
    main()
