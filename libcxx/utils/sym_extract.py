#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##
"""
sym_extract - Extract and output a list of symbols from a shared library.
"""
from argparse import ArgumentParser
from libcxx.sym_check import extract, util


def main():
    parser = ArgumentParser(
        description='Extract a list of symbols from a shared library.')
    parser.add_argument('library', metavar='shared-lib', type=str,
                        help='The library to extract symbols from')
    parser.add_argument('-o', '--output', dest='output',
                        help='The output file. stdout is used if not given',
                        type=str, action='store', default=None)
    parser.add_argument('--names-only', dest='names_only',
                        help='Output only the name of the symbol',
                        action='store_true', default=False)
    parser.add_argument('--only-stdlib-symbols', dest='only_stdlib',
                        help="Filter all symbols not related to the stdlib",
                        action='store_true', default=False)
    parser.add_argument('--defined-only', dest='defined_only',
                        help="Filter all symbols that are not defined",
                        action='store_true', default=False)
    parser.add_argument('--undefined-only', dest='undefined_only',
                        help="Filter all symbols that are defined",
                        action='store_true', default=False)

    args = parser.parse_args()
    assert not (args.undefined_only and args.defined_only)
    if args.output is not None:
        print('Extracting symbols from %s to %s.'
              % (args.library, args.output))
    syms = extract.extract_symbols(args.library)
    if args.only_stdlib:
        syms, other_syms = util.filter_stdlib_symbols(syms)
    filter = lambda x: x
    if args.defined_only:
      filter = lambda l: list([x for x in l if x['is_defined']])
    if args.undefined_only:
      filter = lambda l: list([x for x in l if not x['is_defined']])
    util.write_syms(syms, out=args.output, names_only=args.names_only, filter=filter)


if __name__ == '__main__':
    main()
