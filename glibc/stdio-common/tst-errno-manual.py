#!/usr/bin/python3
# Check is all errno definitions at errlist.h documented in the manual.
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

import argparse
import sys
import re

RE_MANUAL = re.compile(
    r'(?:^@errno){(\w+)')

RE_ERRLIST = re.compile(
    r'\(E[a-zA-Z0-9]+\)')

PASS=0
FAIL=1

# Each manual entry is in the form:
#
#  errno{EAGAIN, 35, Resource temporarily unavailable}
def parse_manual(f):
    errlist = [RE_MANUAL.findall(s) for s in f]
    return map(lambda x : x[0], filter(None, errlist))

# Each errlist entry is in the form:
#
#  _S(ERR_MAP(EAGAIN), N_("Resource temporarily unavailable"))
def parse_errlist(f):
    errlist = [RE_ERRLIST.findall(s) for s in f]
    # Each element is '[]' or '['(EAGAIN)']'
    return map(lambda s : s[0][s[0].find('(')+1:s[0].find(')')],
               filter(None, errlist))

def check_errno_definitions(manual_fname, errlist_fname):
    with open(manual_fname, 'r') as mfile, open(errlist_fname, 'r') as efile:
        merr = parse_manual(mfile)
        eerr = parse_errlist(efile)
        diff = set(eerr).difference(merr)
        if not diff:
            sys.exit(PASS)
        else:
            print("Failure: the following value(s) are not in manual:",
                  ", ".join(str(e) for e in diff))
            sys.exit(FAIL)

def main():
    parser = argparse.ArgumentParser(description='Generate errlist.h')
    parser.add_argument('-m', dest='manual', metavar='FILE',
                        help='manual errno texi file')
    parser.add_argument('-e', dest='errlist', metavar='FILE',
                        help='errlist with errno definitions')
    args = parser.parse_args()

    check_errno_definitions(args.manual, args.errlist)


if __name__ == '__main__':
    main()
