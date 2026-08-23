# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##


import sys

import gdb

test_fail = False
bp_hit = False


def exit_handler(event):
    exit_code = getattr(event, "exit_code", None)

    if not bp_hit or test_fail or exit_code != 0:
        print(
            f"bp_hit: {bp_hit}, test_fail: {test_fail}, exit: {exit_code}",
            file=sys.stderr,
        )
        print("Failed GDB test", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


def bp_handler(event):
    global bp_hit
    global test_fail
    try:
        bp_hit = True

        frame = gdb.newest_frame()
        found_main = False

        while frame != None:
            if frame.name() == "main":
                found_main = True
                break
            frame = frame.older()

        if not found_main:
            test_fail = True

    finally:
        gdb.execute("continue")


def main():
    gdb.execute("set height 0")
    gdb.execute("set python print-stack full")

    gdb.events.stop.connect(bp_handler)
    gdb.events.exited.connect(exit_handler)
    gdb.execute("run")

    print("Inferior didn't exit as expected", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
