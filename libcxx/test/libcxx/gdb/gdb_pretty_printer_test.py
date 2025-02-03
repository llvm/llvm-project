# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""Commands used to automate testing gdb pretty printers.

This script is part of a larger framework to test gdb pretty printers. It
runs the program, detects test cases, checks them, and prints results.

See gdb_pretty_printer_test.sh.cpp on how to write a test case.

"""

from __future__ import print_function
import json
import re
import gdb
import sys

test_failures = 0
# Sometimes the inital run command can fail to trace the process.
# (e.g. you don't have ptrace permissions)
# In these cases gdb still sends us an exited event so we cannot
# see what "run" printed to check for a warning message, since
# we get taken to our exit handler before we can look.
# Instead check that at least one test has been run by the time
# we exit.
has_run_tests = False

has_execute_mi = tuple(map(int, gdb.VERSION.split("."))) >= (14, 2)

class CheckResult(gdb.Command):
    def __init__(self):
        super(CheckResult, self).__init__("print_and_compare", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        global has_run_tests

        try:
            has_run_tests = True

            # Stack frame is:
            # 0. StopForDebugger
            # 1. CompareListChildrenToChars, ComparePrettyPrintToChars or ComparePrettyPrintToRegex
            # 2. TestCase
            compare_frame = gdb.newest_frame().older()
            testcase_frame = compare_frame.older()
            test_loc = testcase_frame.find_sal()
            test_loc_str = test_loc.symtab.filename + ":" + str(test_loc.line)
            # Use interactive commands in the correct context to get the pretty
            # printed version

            frame_name = compare_frame.name()
            if frame_name.startswith("CompareListChildren"):
                if has_execute_mi:
                    value = self._get_children(compare_frame)
                else:
                    print("SKIPPED: " + test_loc_str)
                    return
            else:
                value = self._get_value(compare_frame, testcase_frame)

            gdb.newest_frame().select()
            expectation_val = compare_frame.read_var("expectation")
            check_literal = expectation_val.string(encoding="utf-8")
            if "PrettyPrintToRegex" in frame_name:
                test_fails = not re.search(check_literal, value)
            else:
                test_fails = value != check_literal

            if test_fails:
                global test_failures
                print("FAIL: " + test_loc_str)
                print("GDB printed:")
                print("   " + repr(value))
                print("Value should match:")
                print("   " + repr(check_literal))
                test_failures += 1
            else:
                print("PASS: " + test_loc_str)

        except RuntimeError as e:
            # At this point, lots of different things could be wrong, so don't try to
            # recover or figure it out. Don't exit either, because then it's
            # impossible to debug the framework itself.
            print("FAIL: Something is wrong in the test framework.")
            print(str(e))
            test_failures += 1

    def _get_children(self, compare_frame):
        compare_frame.select()
        gdb.execute_mi("-var-create", "value", "*", "value")
        r = gdb.execute_mi("-var-list-children", "--simple-values", "value")
        gdb.execute_mi("-var-delete", "value")
        children = r["children"]
        if r["displayhint"] == "map":
            r = [
                {
                    "key": json.loads(children[2 * i]["value"]),
                    "value": json.loads(children[2 * i + 1]["value"]),
                }
                for i in range(len(children) // 2)
            ]
        else:
            r = [json.loads(el["value"]) for el in children]
        return json.dumps(r, sort_keys=True)

    def _get_value(self, compare_frame, testcase_frame):
        compare_frame.select()
        frame_name = compare_frame.name()
        if frame_name.startswith("ComparePrettyPrint"):
            s = gdb.execute("p value", to_string=True)
        else:
            value_str = str(compare_frame.read_var("value"))
            clean_expression_str = value_str.strip("'\"")
            testcase_frame.select()
            s = gdb.execute("p " + clean_expression_str, to_string=True)
        if sys.version_info.major == 2:
            s = s.decode("utf-8")

        # Ignore the convenience variable name and newline
        return s[s.find("= ") + 2 : -1]


def exit_handler(event=None):
    global test_failures
    global has_run_tests

    if not has_run_tests:
        print("FAILED test program did not run correctly, check gdb warnings")
        test_failures = -1
    elif test_failures:
        print("FAILED %d cases" % test_failures)
    exit(test_failures)


# Start code executed at load time

# Disable terminal paging
gdb.execute("set height 0")
gdb.execute("set python print-stack full")

if has_execute_mi:
    gdb.execute_mi("-enable-pretty-printing")

test_failures = 0
CheckResult()
test_bp = gdb.Breakpoint("StopForDebugger")
test_bp.enabled = True
test_bp.silent = True
test_bp.commands = "print_and_compare\ncontinue"
# "run" won't return if the program exits; ensure the script regains control.
gdb.events.exited.connect(exit_handler)
gdb.execute("run")
# If the program didn't exit, something went wrong, but we don't
# know what. Fail on exit.
test_failures += 1
exit_handler(None)
