#!/usr/bin/python3
# Check ELF program headers for WX segments.
# Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

"""Check that the program headers do not contain write-exec segments."""

import argparse
import os.path
import re
import sys

# Regular expression to extract the RWE flags field.  The
# address/offset columns have varying width.
RE_LOAD = re.compile(
    r'^  LOAD +(?:0x[0-9a-fA-F]+ +){5}([R ][W ][ E]) +0x[0-9a-fA-F]+\n\Z')

def process_file(path, inp, xfail):
    """Analyze one input file."""

    errors = 0
    for line in inp:
        error = None
        if line.startswith('  LOAD '):
            match = RE_LOAD.match(line)
            if match is None:
                error = 'Invalid LOAD line'
            else:
                flags, = match.groups()
                if 'W' in flags and 'E' in flags:
                    if xfail:
                        print('{}: warning: WX segment (as expected)'.format(
                            path))
                    else:
                        error = 'WX segment'

        if error is not None:
            print('{}: error: {}: {!r}'.format(path, error, line.strip()))
            errors += 1

    if xfail and errors == 0:
        print('{}: warning: missing expected WX segment'.format(path))
    return errors


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xfail',
                        help='Mark input files as XFAILed ("*" for all)',
                        type=str, default='')
    parser.add_argument('phdrs',
                        help='Files containing readelf -Wl output',
                        nargs='*')
    opts = parser.parse_args(sys.argv)

    xfails = set(opts.xfail.split(' '))
    xfails_all = opts.xfail.strip() == '*'

    errors = 0
    for path in opts.phdrs:
        xfail = ((os.path.basename(path) + '.phdrs') in xfails
                 or xfails_all)
        with open(path) as inp:
            errors += process_file(path, inp, xfail)
    if errors > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
