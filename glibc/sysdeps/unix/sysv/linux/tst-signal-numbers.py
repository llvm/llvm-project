#!/usr/bin/python3
# Test that glibc's signal numbers match the kernel's.
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
        description="Test that glibc's signal numbers match the kernel's.")
    parser.add_argument('--cc', metavar='CC',
                        help='C compiler (including options) to use')
    args = parser.parse_args()
    sys.exit(glibcextract.compare_macro_consts(
        '#define _GNU_SOURCE 1\n'
        '#include <signal.h>\n',
        '#define _GNU_SOURCE 1\n'
        '#include <stddef.h>\n'
        '#include <asm/signal.h>\n',
        args.cc,
        # Filter out constants that aren't signal numbers.
        'SIG[A-Z]+',
        # Discard obsolete signal numbers and unrelated constants:
        #    SIGCLD, SIGIOT, SIGSWI, SIGUNUSED.
        #    SIGSTKSZ, SIGRTMIN, SIGRTMAX.
        'SIG(CLD|IOT|RT(MIN|MAX)|STKSZ|SWI|UNUSED)'))

if __name__ == '__main__':
    main()
