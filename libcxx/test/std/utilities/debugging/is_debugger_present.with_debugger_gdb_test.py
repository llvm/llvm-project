# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from __future__ import print_function
import re
import gdb
import sys

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
            print("GDB Custom Test is starting!")

            has_run_tests = True

            # Stack frame is:
            # 0. StopForDebugger
            # 1. ComparePrettyPrintToChars or ComparePrettyPrintToRegex
            # 2. TestCase
            compare_frame = gdb.newest_frame().older()
            testcase_frame = compare_frame.older()
            test_loc = testcase_frame.find_sal()

            # Use interactive commands in the correct context to get the pretty
            # printed version

            value_str = self._get_value_string(compare_frame, testcase_frame)
            print(f"====> GDB output: {compare_frame}")
            print(f"====> GDB output: {testcase_frame}")
            print(f"====> GDB output: {test_loc}")
            print(f"====> GDB output: {value_str}")

            # Ignore the convenience variable name and newline
            value = value_str[value_str.find("= ") + 2 : -1]
            gdb.newest_frame().select()
            expectation_val = compare_frame.read_var("isDebuggerPresent")
            print(f"====> GDB: expectation_val isDebuggerPresent = {expectation_val}")
            if not expectation_val:
                global test_failures
                print("FAIL: " + test_loc.symtab.filename + ":" + str(test_loc.line))
                print("`isDebuggerPresent` value is `false`, value should be `true`")
                test_failures += 1
            else:
                print("PASS: " + test_loc.symtab.filename + ":" + str(test_loc.line))
            
            expectation_val = compare_frame.read_var("isDebuggerPresent1")
            if expectation_val:
                print(f"====> GDB: expectation_val isDebuggerPresent1 = {expectation_val}")
            expectation_val = compare_frame.read_var("helpMeStr")
            if expectation_val == "Yeah it is working!":
                print(f"====> GDB: expectation_val helpMeStr = {expectation_val}")
            else:
                print(f"====> GDB error: helpMeStr  {expectation_val}")

            # check_literal = expectation_val.string(encoding="utf-8")
            # if "PrettyPrintToRegex" in compare_frame.name():
            #     test_fails = not re.search(check_literal, value)
            # else:
            #     test_fails = value != check_literal

            # if test_fails:
            #     global test_failures
            #     print("FAIL: " + test_loc.symtab.filename + ":" + str(test_loc.line))
            #     print("GDB printed:")
            #     print("   " + repr(value))
            #     print("Value should match:")
            #     print("   " + repr(check_literal))
            #     test_failures += 1
            # else:
            #     print("PASS: " + test_loc.symtab.filename + ":" + str(test_loc.line))

        except RuntimeError as e:
            # At this point, lots of different things could be wrong, so don't try to
            # recover or figure it out. Don't exit either, because then it's
            # impossible to debug the framework itself.
            print("FAIL: Something is wrong in the test framework.")
            print(str(e))

            test_failures += 1

    # def _get_value_string(self, compare_frame, testcase_frame):
    #     compare_frame.select()
    #     if "ComparePrettyPrint" in compare_frame.name():
    #         s = gdb.execute("p value", to_string=True)
    #     else:
    #         value_str = str(compare_frame.read_var("value"))
    #         clean_expression_str = value_str.strip("'\"")
    #         testcase_frame.select()
    #         s = gdb.execute("p " + clean_expression_str, to_string=True)
    #     if sys.version_info.major == 2:
    #         return s.decode("utf-8")
    #     return s
    def _get_value_string(self, compare_frame, testcase_frame):
        compare_frame.select()
        # if "ComparePrettyPrint" in compare_frame.name():
        #     s = gdb.execute("p value", to_string=True)
        # else:
        #     value_str = str(compare_frame.read_var("value"))
        #     clean_expression_str = value_str.strip("'\"")
        #     testcase_frame.select()
        #     s = gdb.execute("p " + clean_expression_str, to_string=True)
        # if sys.version_info.major == 2:
        #     return s.decode("utf-8")
        s = compare_frame.name()
        return s


def exit_handler(event=None):
    global test_failures
    global has_run_tests

    if not has_run_tests:
        print("FAILED test program did not run correctly, check gdb warnings")
        test_failures = -1
    elif test_failures:
        print("FAILED %d cases" % test_failures)
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
    # test_bp.commands = "check_is_debugger_present\ncontinue"
    test_bp.commands = """check_is_debugger_present
    continue"""

    # "run" won't return if the program exits; ensure the script regains control.
    gdb.events.exited.connect(exit_handler)
    gdb.execute("run")
    # If the program didn't exit, something went wrong, but we don't
    # know what. Fail on exit.
    test.test_failures += 1
    exit_handler(None)

    print(f"Test failures count: {test.test_failures}")

if __name__ == "__main__":
    main()
