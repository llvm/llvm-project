# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##


import sys

import gdb

stop_event = None


def bp_handler(event):
    global stop_event
    stop_event = event


class TestFail(Exception):
    pass


def main():
    gdb.execute("set height 0")
    gdb.execute("set python print-stack full")
    gdb.execute("set confirm off")
    gdb.events.stop.connect(bp_handler)
    gdb.execute("run")

    try:
        if stop_event is None:
            raise TestFail("Didn't stop at breakpoint")

        frame = gdb.newest_frame()
        found_main = False
        print("Looking for main", file=sys.stderr)
        while frame is not None:
            print(frame.name(), file=sys.stderr)
            if frame.name() == "main":
                found_main = True
                break
            frame = frame.older()

        if not found_main:
            raise TestFail("Could not find main in stopped frames")
    except TestFail as e:
        print(f"Test Failure: {e}", file=sys.stderr)
        gdb.execute("quit 1")

    print("Exiting successfully", file=sys.stderr)
    gdb.execute("quit 0")


if __name__ == "__main__":
    main()
