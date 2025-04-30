#!/usr/bin/python3
# Test that glibc's sys/socket.h SO_* constants match the kernel's.
# Copyright (C) 2018-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

import argparse
import sys

import glibcextract

def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(
        description="Test that glibc's sys/socket.h constants "
        "match the kernel's.")
    parser.add_argument('--cc', metavar='CC',
                        help='C compiler (including options) to use')
    args = parser.parse_args()

    def check(define):
        return glibcextract.compare_macro_consts(
            source_1=define + '#include <sys/socket.h>\n',
            # Some constants in <asm/socket.h> may depend on the size
            # of pid_t or time_t.
            source_2='#include <sys/types.h>\n'
            '#include <asm/socket.h>\n',
            cc=args.cc,
            # We cannot compare all macros because some macros cannot
            # be expanded as constants, and glibcextract currently is
            # not able to isolate errors.
            macro_re='SOL?_.*',
            # <sys/socket.h> and <asm/socket.h> are not a good match.
            # Most socket-related constants are not defined in any
            # UAPI header.  Check only the intersection of the macros
            # in both headers.  Regular tests ensure that expected
            # macros for _GNU_SOURCE are present, and the conformance
            # tests cover most of the other modes.
            allow_extra_1=True,
            allow_extra_2=True)
    # _GNU_SOURCE is defined by include/libc-symbols.h, which is
    # included by the --cc command.  Defining _ISOMAC does not prevent
    # that.
    status = max(
        check(''),
        check('#undef _GNU_SOURCE\n'),
        check('#undef _GNU_SOURCE\n'
              '#define _POSIX_SOURCE 1\n'))
    sys.exit(status)

if __name__ == '__main__':
    main()
