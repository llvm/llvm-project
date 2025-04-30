#!/usr/bin/python3
# Print a list of symbols exported by some headers that would
# otherwise be in the user's namespace.
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

import glibcconform

# Extra symbols possibly not found through -aux-info but still
# reserved by the standard: either data symbols, or symbols where the
# standard leaves unspecified whether the identifier is a macro or
# defined with external linkage.
EXTRA_SYMS = {}
EXTRA_SYMS['ISO'] = {'errno', 'setjmp', 'va_end'}
EXTRA_SYMS['ISO99'] = EXTRA_SYMS['ISO'] | {'math_errhandling'}
# stdatomic.h not yet covered by conformance tests; as per DR#419, all
# the generic functions there or may not be defined with external
# linkage (but are reserved in any case).
EXTRA_SYMS['ISO11'] = EXTRA_SYMS['ISO99']
# The following lists may not be exhaustive.
EXTRA_SYMS['POSIX'] = (EXTRA_SYMS['ISO']
                       | {'environ', 'sigsetjmp', 'optarg', 'optind', 'opterr',
                          'optopt', 'tzname'})
EXTRA_SYMS['XPG4'] = (EXTRA_SYMS['POSIX']
                      | {'signgam', 'loc1', 'loc2', 'locs', 'daylight',
                         'timezone'})
EXTRA_SYMS['XPG42'] = EXTRA_SYMS['XPG4'] | {'getdate_err', 'h_errno'}
EXTRA_SYMS['UNIX98'] = EXTRA_SYMS['XPG42']
EXTRA_SYMS['XOPEN2K'] = (EXTRA_SYMS['POSIX']
                         | {'signgam', 'daylight', 'timezone', 'getdate_err',
                            'h_errno', 'in6addr_any', 'in6addr_loopback'})
EXTRA_SYMS['POSIX2008'] = (EXTRA_SYMS['POSIX']
                           | {'in6addr_any', 'in6addr_loopback'})
EXTRA_SYMS['XOPEN2K8'] = (EXTRA_SYMS['POSIX2008']
                          | {'signgam', 'daylight', 'timezone', 'getdate_err'})


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(description='List exported symbols.')
    parser.add_argument('--headers', metavar='HEADERS',
                        help='list of headers')
    parser.add_argument('--standard', metavar='STD',
                        help='standard to use when processing headers')
    parser.add_argument('--cc', metavar='CC',
                        help='C compiler to use')
    parser.add_argument('--flags', metavar='CFLAGS',
                        help='Compiler flags to use with CC')
    args = parser.parse_args()
    fns = set()
    compiler = '%s %s' % (args.cc, args.flags)
    for header in args.headers.split():
        fns |= glibcconform.list_exported_functions(compiler, args.standard,
                                                    header)
    fns |= EXTRA_SYMS[args.standard]
    print('\n'.join(sorted(fn for fn in fns if not fn.startswith('_'))))


if __name__ == '__main__':
    main()
