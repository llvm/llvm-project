#!/usr/bin/python3
# Check that a wrapper header exist for each non-sysdeps header.
# Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

# Non-sysdeps subdirectories are not on the C include path, so
# installed headers need to have a sysdep wrapper header.
#
# usage: scripts/checl-wrapper-headers.py \
#     --root=$(..) --subdir=$(subdir) $(headers) \
#     [--generated $(common-generated)]
#
# If invoked with --root=., the script is invoked from the root of the
# source tree, so paths starting with "include/" are skipped (because
# those do not require wrappers).

import argparse
import os
import sys

# Some subdirectories are only compiled for essentially one target.
# In this case, we do not need to check for consistent wrapper
# headers.  Hurd uses a custom way to Hurd-specific inject wrapper
# headers; see sysdeps/mach/Makefiles under "ifdef in-Makerules".
SINGLE_TARGET_SUBDIRS = frozenset(("hurd", "mach"))

# Name of the special subdirectory with the wrapper headers.
INCLUDE = "include"

def check_sysdeps_bits(args):
    """Check that the directory sysdeps/generic/bits does not exist."""
    bits = os.path.join(args.root, 'sysdeps', 'generic', 'bits')
    if os.path.exists(bits):
        # See commit c72565e5f1124c2dc72573e83406fe999e56091f and
        # <https://sourceware.org/ml/libc-alpha/2016-05/msg00189.html>.
        print('error: directory {} has been added, use bits/ instead'.format(
            os.path.relpath(os.path.realpath(bits), args.root)))
        return False
    return True

def check_headers_root(args):
    """Check headers located at the top level of the source tree."""
    good = True
    generated = frozenset(args.generated)
    for header in args.headers:
        if not (header.startswith('bits/')
                or os.path.exists(os.path.join(args.root, INCLUDE, header))
                or header in generated):
            print('error: top-level header {} must be in bits/ or {}/'
                  .format(header, INCLUDE))
            good = False
    return good

def check_headers(args):
    """Check headers located in a subdirectory."""
    good = True
    for header in args.headers:
        # Whitelist .x files, which never have include wrappers.
        if header.endswith(".x"):
            continue

        is_nonsysdep_header = os.access(header, os.R_OK)
        if is_nonsysdep_header:
            # Skip Fortran header files.
            if header.startswith("finclude/"):
                continue

            include_path = os.path.join(args.root, INCLUDE, header)
            if not os.access(include_path, os.R_OK):
                print('error: missing wrapper header {} for {}'.format(
                    os.path.join(INCLUDE, header),
                    os.path.relpath(os.path.realpath(header), args.root)))
                good = False
    return good

def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(
        description='Check for missing wrapper headers in include/.')
    parser.add_argument('--root', metavar='DIRECTORY', required=True,
                        help='Path to the top-level of the source tree')
    parser.add_argument('--subdir', metavar='DIRECTORY', required=True,
                        help='Name of the subdirectory being processed')
    parser.add_argument('--generated', metavar='FILE', default="", nargs="*",
                        help="Generated files (which are ignored)")
    parser.add_argument('headers', help='Header files to process', nargs='+')
    args = parser.parse_args()

    good = (args.root == '.') == (args.subdir == '.')
    if not good:
        print('error: --root/--subdir disagree about top-of-tree location')

    if args.subdir == '.':
        good &= check_sysdeps_bits(args)
        good &= check_headers_root(args)
    elif args.subdir not in SINGLE_TARGET_SUBDIRS:
        good &= check_headers(args)

    if not good:
        sys.exit(1)

if __name__ == '__main__':
    main()
