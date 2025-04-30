#!/usr/bin/python3
# Move symbols from other shared objects into libc.so.
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

"""Move symbols from other shared objects into libc.so.

This script moves ABI symbols from non-libc abilists in to
libc.abilist.  Symbol versions are preserved.  The script must be
called from the top of the glibc source tree.

"""

import argparse
import os.path
import sys

# Make available glibc Python modules.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import glibcsymbols

def add_to_libc_path(path, new_symbols):
    """Add SYMBOLS to the abilist file PATH.

    NEW_SYMBOLS is a dictionary from glibcsymbols.VersionedSymbol
    objects to their flags.

    """
    original_symbols = glibcsymbols.read_abilist(path)
    updated_symbols = original_symbols.copy()
    updated_symbols.update(new_symbols)
    if updated_symbols != original_symbols:
        sys.stdout.write('updating libc abilist {}\n'.format(path))
        glibcsymbols.replace_file(
            path, glibcsymbols.abilist_lines(updated_symbols))

# The name of the libc.so abilist file.
libc_abilist = 'libc.abilist'

def add_to_libc_fallback(directory, subdirs, symbol_lines):
    """Add SYMBOL_LINES to the libc.abilist files in SUBDIRS in DIRECTORY.

    All subdirectories must exist.  If they do, return True.  If not,
    skip processing and return False.

    """
    abilists = [os.path.join(directory, subdir, libc_abilist)
                for subdir in subdirs]
    for abilist in abilists:
        if not os.path.exists(abilist):
            return False
    for abilist in abilists:
            add_to_libc_path(abilist, symbol_lines)
    return True

def add_to_libc(directory, symbol_lines):

    """Add SYMBOL_LINES (a list of strings) to libc.abilist in DIRECTORY.

    Try specific subdirectories as well if libc.abilist is not found
    in DIRECTORY.

    """
    libc_path = os.path.join(directory, libc_abilist)
    if os.path.exists(libc_path):
        add_to_libc_path(libc_path, symbol_lines)
        return

    # Special case for powerpc32 and mips32 variants.
    if add_to_libc_fallback(directory, ('fpu', 'nofpu'), symbol_lines):
        return

    # Special case for mips64.
    if add_to_libc_fallback(directory, ('n32', 'n64'), symbol_lines):
        return

    raise IOError('No libc.abilist found for: {}'.format(directory))

def move_symbols_1(path, to_move, moved_symbols):
    """Move SYMBOLS from the abilist file PATH to MOVED_SYMBOLS.

    TO_MOVE must be a set of strings.  MOVED_SYMBOLS is a dictionary.

    """
    suffix = '.abilist'
    assert path.endswith('.abilist')
    library = os.path.basename(path)[:-len(suffix)]
    placeholder = '__{}_version_placeholder'.format(library)

    new_lines = []
    changed = False

    old_symbols = glibcsymbols.read_abilist(path)
    old_versions = set(versym.version for versym in old_symbols.keys())
    matching_symbols = dict(e for e in old_symbols.items()
                            if e[0].symbol in to_move)
    if matching_symbols:
        sys.stdout.write('updating {} abilist {}\n'.format(library, path))
        new_symbols = dict(e for e in old_symbols.items()
                           if e[0].symbol not in to_move)

        # Add placeholder symbols to prevent symbol versions from
        # going away completely.
        new_versions = set(versym.version for versym in new_symbols.keys())
        for missing_version in old_versions - new_versions:
            new_symbols[glibcsymbols.VersionedSymbol(
                placeholder, missing_version)] = 'F'

        glibcsymbols.replace_file(
            path, glibcsymbols.abilist_lines(new_symbols))

        moved_symbols.update(matching_symbols)

def move_symbols(directory, files, symbols):
    """Move SYMBOLS from FILES (a list of abilist file names) in DIRECTORY.

    SYMBOLS must be a set of strings.

    """
    moved_symbols = {}
    for filename in files:
        move_symbols_1(os.path.join(directory, filename), symbols,
                       moved_symbols)
    if moved_symbols:
        add_to_libc(directory, moved_symbols)

def get_parser():
    """Return an argument parser for this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-linux', action='store_true',
                        help='Restrict the operation to Linux abilists')
    parser.add_argument('symbols', help='name of the symbol to move',
                        nargs='+')
    return parser

def main(argv):
    """The main entry point."""
    parser = get_parser()
    opts = parser.parse_args(argv)
    if opts.only_linux:
        sysdeps = 'sysdeps/unix/sysv/linux'
    else:
        sysdeps = 'sysdeps'

    symbols = frozenset(opts.symbols)

    for directory, dirs, files in os.walk(sysdeps):
        move_symbols(directory, [name for name in files
                                 if name != 'libc.abilist'
                                 and name.endswith('.abilist')], symbols)

if __name__ == '__main__':
    main(sys.argv[1:])
