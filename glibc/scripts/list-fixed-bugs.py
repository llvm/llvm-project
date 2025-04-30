#!/usr/bin/python3
# Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

"""List fixed bugs for the NEWS file.

This script takes a version number as input and generates a list of
bugs marked as FIXED with that milestone, to be added to the NEWS file
just before release.  The output is in UTF-8.
"""

import argparse
import json
import sys
import textwrap
import urllib.request


def get_parser():
    """Return an argument parser for this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version',
                        help='Release version to look up')
    return parser


def list_fixed_bugs(version):
    """List the bugs fixed in a given version."""
    url = ('https://sourceware.org/bugzilla/rest.cgi/bug?product=glibc'
           '&resolution=FIXED&target_milestone=%s'
           '&include_fields=id,component,summary' % version)
    response = urllib.request.urlopen(url)
    json_data = response.read().decode('utf-8')
    data = json.loads(json_data)
    for bug in data['bugs']:
        desc = '[%d] %s: %s' % (bug['id'], bug['component'], bug['summary'])
        desc = textwrap.fill(desc, width=72, initial_indent='  ',
                             subsequent_indent='    ') + '\n'
        sys.stdout.buffer.write(desc.encode('utf-8'))


def main(argv):
    """The main entry point."""
    parser = get_parser()
    opts = parser.parse_args(argv)
    list_fixed_bugs(opts.version)


if __name__ == '__main__':
    main(sys.argv[1:])
