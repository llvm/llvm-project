#!/usr/bin/python3
# Check that use of symbols declared in a given header does not result
# in any symbols being brought in that are not reserved with external
# linkage for the given standard.
# Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
from collections import defaultdict
import os.path
import re
import subprocess
import sys
import tempfile

import glibcconform

# The following whitelisted symbols are also allowed for now.
#
# * Bug 17576: stdin, stdout, stderr only reserved with external
# linkage when stdio.h included (and possibly not then), not
# generally.
#
# * Bug 18442: re_syntax_options wrongly brought in by regcomp and
# used by re_comp.
#
WHITELIST = {'stdin', 'stdout', 'stderr', 're_syntax_options'}


def list_syms(filename):
    """Return information about GLOBAL and WEAK symbols listed in readelf
    -s output."""
    ret = []
    cur_file = filename
    with open(filename, 'r') as syms_file:
        for line in syms_file:
            line = line.rstrip()
            if line.startswith('File: '):
                cur_file = line[len('File: '):]
                cur_file = cur_file.split('/')[-1]
                continue
            # Architecture-specific st_other bits appear inside [] and
            # disrupt the format of readelf output.
            line = re.sub(r'\[.*?\]', '', line)
            fields = line.split()
            if len(fields) < 8:
                continue
            bind = fields[4]
            ndx = fields[6]
            sym = fields[7]
            if bind not in ('GLOBAL', 'WEAK'):
                continue
            if not re.fullmatch('[A-Za-z0-9_]+', sym):
                continue
            ret.append((cur_file, sym, bind, ndx != 'UND'))
    return ret


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(description='Check link-time namespace.')
    parser.add_argument('--header', metavar='HEADER',
                        help='name of header')
    parser.add_argument('--standard', metavar='STD',
                        help='standard to use when processing header')
    parser.add_argument('--cc', metavar='CC',
                        help='C compiler to use')
    parser.add_argument('--flags', metavar='CFLAGS',
                        help='Compiler flags to use with CC')
    parser.add_argument('--stdsyms', metavar='FILE',
                        help='File with list of standard symbols')
    parser.add_argument('--libsyms', metavar='FILE',
                        help='File with symbol information from libraries')
    parser.add_argument('--readelf', metavar='READELF',
                        help='readelf program to use')
    args = parser.parse_args()

    # Load the list of symbols that are OK.
    stdsyms = set()
    with open(args.stdsyms, 'r') as stdsyms_file:
        for line in stdsyms_file:
            stdsyms.add(line.rstrip())
    stdsyms |= WHITELIST

    # Load information about GLOBAL and WEAK symbols defined or used
    # in the standard libraries.
    # Symbols from a given object, except for weak defined symbols.
    seen_syms = defaultdict(list)
    # Strong undefined symbols from a given object.
    strong_undef_syms = defaultdict(list)
    # Objects defining a given symbol (strongly or weakly).
    sym_objs = defaultdict(list)
    for file, name, bind, defined in list_syms(args.libsyms):
        if defined:
            sym_objs[name].append(file)
        if bind == 'GLOBAL' or not defined:
            seen_syms[file].append(name)
        if bind == 'GLOBAL' and not defined:
            strong_undef_syms[file].append(name)

    # Determine what ELF-level symbols are brought in by use of C-level
    # symbols declared in the given header.
    #
    # The rules followed are heuristic and so may produce false
    # positives and false negatives.
    #
    # * All undefined symbols are considered of signficance, but it is
    # possible that (a) any standard library definition is weak, so
    # can be overridden by the user's definition, and (b) the symbol
    # is only used conditionally and not if the program is limited to
    # standard functionality.
    #
    # * If a symbol reference is only brought in by the user using a
    # data symbol rather than a function from the standard library,
    # this will not be detected.
    #
    # * If a symbol reference is only brought in by crt*.o or libgcc,
    # this will not be detected.
    #
    # * If a symbol reference is only brought in through __builtin_foo
    # in a standard macro being compiled to call foo, this will not be
    # detected.
    #
    # * Header inclusions should be compiled several times with
    # different options such as -O2, -D_FORTIFY_SOURCE and
    # -D_FILE_OFFSET_BITS=64 to find out what symbols are undefined
    # from such a compilation; this is not yet implemented.
    #
    # * This script finds symbols referenced through use of macros on
    # the basis that if a macro calls an internal function, that
    # function must also be declared in the header.  However, the
    # header might also declare implementation-namespace functions
    # that are not called by any standard macro in the header,
    # resulting in false positives for any symbols brought in only
    # through use of those implementation-namespace functions.
    #
    # * Namespace issues can apply for dynamic linking as well as
    # static linking, when a call is from one shared library to
    # another or uses a PLT entry for a call within a shared library;
    # such issues are only detected by this script if the same
    # namespace issue applies for static linking.
    seen_where = {}
    files_seen = set()
    all_undef = {}
    current_undef = {}
    compiler = '%s %s' % (args.cc, args.flags)
    c_syms = glibcconform.list_exported_functions(compiler, args.standard,
                                                  args.header)
    with tempfile.TemporaryDirectory() as temp_dir:
        cincfile_name = os.path.join(temp_dir, 'undef.c')
        cincfile_o_name = os.path.join(temp_dir, 'undef.o')
        cincfile_sym_name = os.path.join(temp_dir, 'undef.sym')
        cincfile_text = ('#include <%s>\n%s\n'
                         % (args.header,
                            '\n'.join('void *__glibc_test_%s = (void *) &%s;'
                                      % (sym, sym) for sym in sorted(c_syms))))
        with open(cincfile_name, 'w') as cincfile:
            cincfile.write(cincfile_text)
        cmd = ('%s %s -D_ISOMAC %s -c %s -o %s'
               % (args.cc, args.flags, glibcconform.CFLAGS[args.standard],
                  cincfile_name, cincfile_o_name))
        subprocess.check_call(cmd, shell=True)
        cmd = ('LC_ALL=C %s -W -s %s > %s'
               % (args.readelf, cincfile_o_name, cincfile_sym_name))
        subprocess.check_call(cmd, shell=True)
        for file, name, bind, defined in list_syms(cincfile_sym_name):
            if bind == 'GLOBAL' and not defined:
                sym_text = '[initial] %s' % name
                seen_where[name] = sym_text
                all_undef[name] = sym_text
                current_undef[name] = sym_text

    while current_undef:
        new_undef = {}
        for sym, cu_sym in sorted(current_undef.items()):
            for file in sym_objs[sym]:
                if file in files_seen:
                    continue
                files_seen.add(file)
                for ssym in seen_syms[file]:
                    if ssym not in seen_where:
                        seen_where[ssym] = ('%s -> [%s] %s'
                                            % (cu_sym, file, ssym))
                for usym in strong_undef_syms[file]:
                    if usym not in all_undef:
                        usym_text = '%s -> [%s] %s' % (cu_sym, file, usym)
                        all_undef[usym] = usym_text
                        new_undef[usym] = usym_text
        current_undef = new_undef

    ret = 0
    for sym in sorted(seen_where):
        if sym.startswith('_'):
            continue
        if sym in stdsyms:
            continue
        print(seen_where[sym])
        ret = 1
    sys.exit(ret)


if __name__ == '__main__':
    main()
