# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##


import sys

import gdb


class TestFail(Exception):
    pass


def bp_handler(event):
    try:
        frame = gdb.newest_frame()
        found_main = False

        while frame is not None:
            if frame.name() == "main":
                found_main = True
                break
            frame = frame.older()

        if not found_main:
            raise TestFail("Could not find main in stopped frames")

        gdb.execute("quit 0")
    except TestFail as e:
        print(e, file=sys.stderr)
        gdb.execute("quit 1")


def main():
    gdb.execute("set height 0")
    gdb.execute("set python print-stack full")
    gdb.execute("set confirm off")

    gdb.events.stop.connect(bp_handler)
    gdb.execute("run")

    print("Should have quit by now", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
