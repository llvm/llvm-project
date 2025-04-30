#!/usr/bin/python3
# Produce headers of assembly constants from C expressions.
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

# The input to this script looks like:
#       #cpp-directive ...
#       NAME1
#       NAME2 expression ...
# A line giving just a name implies an expression consisting of just that name.

import argparse

import glibcextract


def gen_test(sym_data):
    """Generate a test for the values of some C constants.

    The first argument is as for glibcextract.compute_c_consts.

    """
    out_lines = []
    for arg in sym_data:
        if isinstance(arg, str):
            if arg == 'START':
                out_lines.append('#include <stdint.h>\n'
                                 '#include <stdio.h>\n'
                                 '#include <bits/wordsize.h>\n'
                                 '#if __WORDSIZE == 64\n'
                                 'typedef uint64_t c_t;\n'
                                 '# define U(n) UINT64_C (n)\n'
                                 '#else\n'
                                 'typedef uint32_t c_t;\n'
                                 '# define U(n) UINT32_C (n)\n'
                                 '#endif\n'
                                 'static int\n'
                                 'do_test (void)\n'
                                 '{\n'
                                 # Compilation test only, using static
                                 # assertions.
                                 '  return 0;\n'
                                 '}\n'
                                 '#include <support/test-driver.c>')
            else:
                out_lines.append(arg)
            continue
        name = arg[0]
        value = arg[1]
#        out_lines.append('_Static_assert (U (asconst_%s) == (c_t) (%s), '
#                         '"value of %s");'
#                         % (name, value, name))
    return '\n'.join(out_lines)


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(
        description='Produce headers of assembly constants.')
    parser.add_argument('--cc', metavar='CC',
                        help='C compiler (including options) to use')
    parser.add_argument('--test', action='store_true',
                        help='Generate test case instead of header')
    parser.add_argument('--python', action='store_true',
                        help='Generate Python file instead of header')
    parser.add_argument('sym_file',
                        help='.sym file to process')
    args = parser.parse_args()
    sym_data = []
    with open(args.sym_file, 'r') as sym_file:
        started = False
        for line in sym_file:
            line = line.strip()
            if line == '':
                continue
            # Pass preprocessor directives through.
            if line.startswith('#'):
                sym_data.append(line)
                continue
            words = line.split(maxsplit=1)
            if not started:
                sym_data.append('START')
                started = True
            # Separator.
            if words[0] == '--':
                continue
            name = words[0]
            value = words[1] if len(words) > 1 else words[0]
            sym_data.append((name, value))
        if not started:
            sym_data.append('START')
    if args.test:
        print(gen_test(sym_data))
    elif args.python:
        consts = glibcextract.compute_c_consts(sym_data, args.cc)
        print('# GENERATED FILE\n'
              '\n'
              '# Constant definitions.\n'
              '# See gen-as-const.py for details.\n')
        print(''.join('%s = %s\n' % c for c in sorted(consts.items())), end='')
    else:
        consts = glibcextract.compute_c_consts(sym_data, args.cc)
        print(''.join('#define %s %s\n' % c for c in sorted(consts.items())), end='')

if __name__ == '__main__':
    main()
