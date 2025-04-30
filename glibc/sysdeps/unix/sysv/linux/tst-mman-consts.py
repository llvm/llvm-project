#!/usr/bin/python3
# Test that glibc's sys/mman.h constants match the kernel's.
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
import glibcsyscalls


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(
        description="Test that glibc's sys/mman.h constants "
        "match the kernel's.")
    parser.add_argument('--cc', metavar='CC',
                        help='C compiler (including options) to use')
    args = parser.parse_args()
    linux_version_headers = glibcsyscalls.linux_kernel_version(args.cc)
    linux_version_glibc = (5, 13)
    sys.exit(glibcextract.compare_macro_consts(
        '#define _GNU_SOURCE 1\n'
        '#include <sys/mman.h>\n',
        '#define _GNU_SOURCE 1\n'
        '#include <linux/mman.h>\n',
        args.cc,
        'MAP_.*',
        # A series of MAP_HUGE_<size> macros are defined by the kernel
        # but not by glibc.  MAP_UNINITIALIZED is kernel-only.
        # MAP_FAILED is not a MAP_* flag and is glibc-only, as is the
        # MAP_ANON alias for MAP_ANONYMOUS.  MAP_RENAME, MAP_AUTOGROW,
        # MAP_LOCAL and MAP_AUTORSRV are in the kernel header for
        # MIPS, marked as "not used by linux"; SPARC has MAP_INHERIT
        # in the kernel header, but does not use it.
        'MAP_HUGE_[0-9].*|MAP_UNINITIALIZED|MAP_FAILED|MAP_ANON'
        '|MAP_RENAME|MAP_AUTOGROW|MAP_LOCAL|MAP_AUTORSRV|MAP_INHERIT',
        linux_version_glibc > linux_version_headers,
        linux_version_headers > linux_version_glibc))

if __name__ == '__main__':
    main()
