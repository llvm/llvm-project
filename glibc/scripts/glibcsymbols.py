#!/usr/bin/python3
# Processing of symbols and abilist files.
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

"""Symbol processing for glibc."""

import os

def replace_file(path, new_contents):
    """Atomically replace PATH with lines from NEW_CONTENTS.

    NEW_CONTENTS must be a sequence of strings.

    """
    temppath = path + 'T'
    with open(temppath, 'w') as out:
        for line in new_contents:
            out.write(line)
    os.rename(temppath, path)

class VersionedSymbol:
    """A combination of a symbol and its version."""

    def __init__(self, symbol, version):
        """Construct a new versioned symbol."""
        assert symbol
        assert version
        self.symbol = symbol
        self.version = version

    def __str__(self):
        return self.symbol + '@' + self.version

    def __eq__(self, other):
        return self.symbol == other.symbol and self.version == other.version

    def __hash__(self):
        return hash(self.symbol) ^ hash(self.version)

def read_abilist(path):
    """Read the abilist file at PATH.

    Return a dictionary from VersionedSymbols to their flags (as
    strings).

    """
    result = {}
    with open(path) as inp:
        for line in inp:
            version, symbol, flags = line.strip().split(' ', 2)
            versym = VersionedSymbol(symbol, version)
            if versym in result:
                raise IOError("{}: duplicate symbol {}".format(path, versym))
            result[versym] = flags
    return result

def abilist_lines(symbols):
    """Build the abilist file contents (as a list of lines).

    SYMBOLS is a dictionary from VersionedSymbols to their flags.

    """
    result = []
    for versym, flags in symbols.items():
        result.append('{} {} {}\n'.format(
            versym.version, versym.symbol, flags))
    result.sort()
    return result
