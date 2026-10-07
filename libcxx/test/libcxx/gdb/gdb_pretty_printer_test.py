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
import os
import re
import tempfile
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


# Parser for GDB<14.2. Expected input formats:
# ^done
# ^done,numchild="1",children=[child={name="value.private",exp="private",numchild="1",value="",thread-id="1"}],has_more="0"
# ^error,msg="Undefined MI command: rubbish"
# See https://sourceware.org/gdb/current/onlinedocs/gdb.html/GDB_002fMI-Result-Records.html
def _parse_mi_record_legacy(rec):
    m = re.match(r"^\^([a-z]+)(?:,(.*))?$", rec)
    if not m:
        raise gdb.error("Failed to parse MI result line: " + rec)
    code, rest = m.group(1), m.group(2)
    if not rest:
        return (code, {})
    s = rest
    idx, L = 0, len(s)

    def skip():
        nonlocal idx
        while idx < L and s[idx].isspace():
            idx += 1

    def iden():
        nonlocal idx
        start = idx
        while idx < L and re.match(r"[A-Za-z0-9_.-]", s[idx]):
            idx += 1
        return s[start:idx]

    def parse_str():
        nonlocal idx
        idx += 1
        out = []
        while idx < L:
            ch = s[idx]
            if ch == '"':
                idx += 1
                return "".join(out)
            if ch == "\\":
                idx += 1
                if idx >= L:
                    break
                e = s[idx]
                idx += 1
                out.append(
                    {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(e, e)
                )
            else:
                out.append(ch)
                idx += 1
        return "".join(out)

    def value():
        nonlocal idx
        skip()
        if idx >= L:
            return None
        ch = s[idx]
        if ch == '"':
            return parse_str()
        if ch == "{":
            return parse_tuple()
        if ch == "[":
            return parse_array()
        raise gdb.error(
            "Unexpected characted {} while parsing MI result line: {}".format(ch, rec)
        )

    def parse_tuple():
        nonlocal idx
        idx += 1
        res = {}
        skip()
        while idx < L and s[idx] != "}":
            k = iden()
            if idx < L and s[idx] == "=":
                idx += 1
                res[k] = value()
            else:
                v = value()
                res[k or str(len(res))] = v
            skip()
            if idx < L and s[idx] == ",":
                idx += 1
            skip()
        if idx < L and s[idx] == "}":
            idx += 1
        return res

    def parse_array():
        nonlocal idx
        idx += 1
        arr = []
        skip()
        while idx < L and s[idx] != "]":
            save = idx
            k = iden()
            if k and idx < L and s[idx] == "=":
                idx += 1
                arr.append({k: value()})
            else:
                idx = save
                arr.append(value())
            skip()
            if idx < L and s[idx] == ",":
                idx += 1
            skip()
        if idx < L and s[idx] == "]":
            idx += 1
        if arr and all(isinstance(x, dict) and len(x) == 1 for x in arr):
            keys = [next(iter(x)) for x in arr]
            if all(k == keys[0] for k in keys):
                arr = [x[keys[0]] for x in arr]
        return arr

    res = {}
    while idx < L:
        skip()
        n = iden()
        if not n:
            break
        if idx < L and s[idx] == "=":
            idx += 1
            res[n] = value()
        else:
            res[n] = value()
        skip()
        if idx < L and s[idx] == ",":
            idx += 1
    return (code, res)


def execute_mi(*args, collect_output=False):
    # gdb.execute_mi is available in GDB 14.2 or later
    if hasattr(gdb, "execute_mi"):
        r = gdb.execute_mi(*args)
        return r if collect_output else None

    # for older GDB: call `interpreter-exec mi2 "-cmd args..."` and parse result like:
    # ^done,numchild="1",children=[child={name="value.private" ...
    mi_command = " ".join(args)
    gdb_command = 'interpreter-exec mi2 "{}"'.format(" ".join(args))

    if not collect_output:
        gdb.execute(gdb_command)
        return

    # gdb.execute("interpreter-exec mi2 ...") ignores flag to_string=True:
    # see https://sourceware.org/bugzilla/show_bug.cgi?id=12886
    # To get output of MI command, we use temporary file.
    # "interpreter-exec mi2" also ignores "set logging file ...".
    # It only prints to stdout, so we:
    # 1) flush the existing stdout
    # 2) redirect the stdout to our temporary file
    # 3) execute the MI command and flush gdb output
    # 4) restore the original stdout file descriptor
    # 5) rewind and read our temporary file
    result_line = ""
    with tempfile.NamedTemporaryFile(mode="w+") as tmp_file:
        stdout_fd = sys.__stdout__.fileno()
        saved_stdout_fd = os.dup(stdout_fd)

        try:
            sys.__stdout__.flush()
            os.dup2(tmp_file.fileno(), stdout_fd)
            gdb.execute(gdb_command)
        finally:
            gdb.flush(gdb.STDOUT)
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)

        tmp_file.seek(0)
        result_line = tmp_file.read().splitlines()[0]

    if not result_line:
        raise gdb.error("No GDB/MI output for: " + mi_command)

    match = re.match(r"\^([a-zA-Z-]+)(?:,(.*))?$", result_line)

    if not match:
        raise gdb.error("Failed to parse MI result line: " + result_line)

    code, payload = _parse_mi_record_legacy(result_line)

    if code == "error":
        msg = payload.get("msg", "Unknown MI error")
        raise gdb.error("GDB/MI Error: " + msg)

    return payload


def execute_expression(expression):
    r = execute_mi("-data-evaluate-expression " + expression)
    return r["value"]


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
                value = self._get_children(compare_frame)
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
        execute_mi("-var-create", "value", "*", "value")
        r = execute_mi(
            "-var-list-children", "--simple-values", "value", collect_output=True
        )
        execute_mi("-var-delete", "value")
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

execute_mi("-enable-pretty-printing")

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
