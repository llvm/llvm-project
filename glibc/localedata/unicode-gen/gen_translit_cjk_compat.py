#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Generate a translit_cjk_compat file from a UnicodeData file.
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

'''
Generate a translit_cjk_compat file from UnicodeData.txt

To see how this script is used, call it with the “-h” option:

    $ ./gen_translit_cjk_compat -h
    … prints usage message …
'''

import argparse
import time
import sys
import unicode_utils

def read_input_file(filename):
    '''Reads the original glibc translit_cjk_compat file to get the
    original head and tail.

    We want to replace only the part of the file between
    “translit_start” and “translit_end”
    '''
    head = tail = ''
    with open(filename, mode='r') as translit_file:
        for line in translit_file:
            head = head + line
            if line.startswith('translit_start'):
                break
        for line in translit_file:
            if line.startswith('translit_end'):
                tail = line
                break
        for line in translit_file:
            tail = tail + line
    return (head, tail)

def output_head(translit_file, unicode_version, head=''):
    '''Write the header of the output file, i.e. the part of the file
    before the “translit_start” line.
    '''
    if ARGS.input_file and head:
        translit_file.write(head)
    else:
        translit_file.write('escape_char /\n')
        translit_file.write('comment_char %\n')
        translit_file.write(unicode_utils.COMMENT_HEADER)
        translit_file.write('\n')
        translit_file.write('% Transliterations of CJK compatibility ')
        translit_file.write('characters.\n')
        translit_file.write('% Generated automatically from UnicodeData.txt '
                            + 'by gen_translit_cjk_compat.py '
                            + 'on {:s} '.format(time.strftime('%Y-%m-%d'))
                            + 'for Unicode {:s}.\n'.format(unicode_version))
        translit_file.write('\n')
        translit_file.write('LC_CTYPE\n')
        translit_file.write('\n')
        translit_file.write('translit_start\n')

def output_tail(translit_file, tail=''):
    '''Write the tail of the output file'''
    if ARGS.input_file and tail:
        translit_file.write(tail)
    else:
        translit_file.write('translit_end\n')
        translit_file.write('\n')
        translit_file.write('END LC_CTYPE\n')

def special_decompose(code_point_list):
    '''
    Decompositions which are not in UnicodeData.txt at all but which
    were used in the original translit_cjk_compat file in glibc and
    which seem to make sense.  I want to keep the update of
    translit_cjk_compat close to the spirit of the original file,
    therefore I added this special decomposition rules here.
    '''
    special_decompose_dict = {
        (0x2215,): [0x002F], # ∕ → /
        (0x00B2,): [0x005E, 0x0032], # ² → ^2
        (0x03BC,): [0x00B5], # μ → µ (GREEK SMALL LETTER MU → MICRO SIGN)
        (0x2113,): [0x006C], # ℓ → l
        (0x00B3,): [0x005E, 0x0033], # ³ → ^3
        (0x00B5,): [0x0075], # µ → u
        (0x03BC, 0x2113): [0x03BC, 0x006C], # μℓ → μl
        (0x0072, 0x0061, 0x0064, 0x2215, 0x0073, 0x00B2): [
            0x0072, 0x0061, 0x0064, 0x002F, 0x0073, 0x00B2],
        (0x006D, 0x2215, 0x0073, 0x00B2): [0x006D, 0x002F, 0x0073, 0x00B2],
    }
    if tuple(code_point_list) in special_decompose_dict:
        return special_decompose_dict[tuple(code_point_list)]
    else:
        return code_point_list

def output_transliteration(translit_file):
    '''Write the new transliteration to the output file'''
    translit_file.write('\n')
    for code_point in sorted(unicode_utils.UNICODE_ATTRIBUTES):
        name = unicode_utils.UNICODE_ATTRIBUTES[code_point]['name']
        decomposition = unicode_utils.UNICODE_ATTRIBUTES[
            code_point]['decomposition']
        if decomposition.startswith('<square>'):
            decomposition = decomposition[9:]
            decomposed_code_points = [[int(x, 16)
                                       for x in decomposition.split(' ')]]
            if decomposed_code_points[0]:
                while True:
                    special_decomposed_code_points = special_decompose(
                        decomposed_code_points[-1])
                    if (special_decomposed_code_points
                            != decomposed_code_points[-1]):
                        decomposed_code_points.append(
                            special_decomposed_code_points)
                        continue
                    special_decomposed_code_points = []
                    for decomposed_code_point in decomposed_code_points[-1]:
                        special_decomposed_code_points += special_decompose(
                            [decomposed_code_point])
                    if (special_decomposed_code_points
                            == decomposed_code_points[-1]):
                        break
                    decomposed_code_points.append(
                        special_decomposed_code_points)
                translit_file.write('% {:s}\n'.format(name))
                translit_file.write('{:s} '.format(
                    unicode_utils.ucs_symbol(code_point)))
                for index in range(0, len(decomposed_code_points)):
                    if index > 0:
                        translit_file.write(';')
                    if len(decomposed_code_points[index]) > 1:
                        translit_file.write('"')
                    for decomposed_code_point in decomposed_code_points[index]:
                        translit_file.write('{:s}'.format(
                            unicode_utils.ucs_symbol(decomposed_code_point)))
                    if len(decomposed_code_points[index]) > 1:
                        translit_file.write('"')
                translit_file.write('\n')
    for code_point in sorted(unicode_utils.UNICODE_ATTRIBUTES):
        name = unicode_utils.UNICODE_ATTRIBUTES[code_point]['name']
        decomposition = unicode_utils.UNICODE_ATTRIBUTES[
            code_point]['decomposition']
        if decomposition and name.startswith('CJK COMPATIBILITY IDEOGRAPH'):
            decomposed_code_points = [int(x, 16)
                                      for x in decomposition.split(' ')]
            if len(decomposed_code_points) != 1:
                sys.stderr.write(
                    'Unexpected decomposition length {:x} {:s} {:s}\n'.format(
                        code_point, name, decomposition))
                exit(1)
            translit_file.write('% {:s}\n'.format(name))
            translit_file.write('{:s} '.format(
                unicode_utils.ucs_symbol(code_point)))
            for decomposed_code_point in decomposed_code_points:
                translit_file.write('{:s}'.format(
                    unicode_utils.ucs_symbol(decomposed_code_point)))
            translit_file.write('\n')
    translit_file.write('\n')

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='''
        Generate a translit_cjk_compat file from UnicodeData.txt.
        ''')
    PARSER.add_argument(
        '-u', '--unicode_data_file',
        nargs='?',
        type=str,
        default='UnicodeData.txt',
        help=('The UnicodeData.txt file to read, '
              + 'default: %(default)s'))
    PARSER.add_argument(
        '-i', '--input_file',
        nargs='?',
        type=str,
        help=''' The original glibc/localedata/locales/translit_cjk_compat
        file.''')
    PARSER.add_argument(
        '-o', '--output_file',
        nargs='?',
        type=str,
        default='translit_cjk_compat.new',
        help='''The new translit_cjk_compat file, default: %(default)s.  If the
        original glibc/localedata/locales/translit_cjk_compat file has
        been given as an option, the header up to the
        “translit_start” line and the tail from the “translit_end”
        line to the end of the file will be copied unchanged into the
        output file.  ''')
    PARSER.add_argument(
        '--unicode_version',
        nargs='?',
        required=True,
        type=str,
        help='The Unicode version of the input files used.')
    ARGS = PARSER.parse_args()

    unicode_utils.fill_attributes(ARGS.unicode_data_file)
    HEAD = TAIL = ''
    if ARGS.input_file:
        (HEAD, TAIL) = read_input_file(ARGS.input_file)
    with open(ARGS.output_file, mode='w') as TRANSLIT_FILE:
        output_head(TRANSLIT_FILE, ARGS.unicode_version, head=HEAD)
        output_transliteration(TRANSLIT_FILE)
        output_tail(TRANSLIT_FILE, tail=TAIL)
