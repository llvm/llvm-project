#!/usr/bin/python3
# -*- coding: utf-8 -*-
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

'''
This script is useful for checking backward compatibility of newly
generated UTF-8 file from utf8_gen.py script

To see how this script is used, call it with the “-h” option:

    $ ./utf8_compatibility.py -h
    … prints usage message …
'''

import sys
import re
import argparse
import unicode_utils

def create_charmap_dictionary(file_name):
    '''Create a dictionary for all code points found in the CHARMAP
    section of a file
    '''
    with open(file_name, mode='r') as utf8_file:
        charmap_dictionary = {}
        for line in utf8_file:
            if line.startswith('CHARMAP'):
                break
        for line in utf8_file:
            if line.startswith('END CHARMAP'):
                return charmap_dictionary
            if line.startswith('%'):
                continue
            match = re.match(
                r'^<U(?P<codepoint1>[0-9A-F]{4,8})>'
                +r'(:?\.\.<U(?P<codepoint2>[0-9-A-F]{4,8})>)?'
                +r'\s+(?P<hexutf8>(/x[0-9a-f]{2}){1,4})',
                line)
            if not match:
                continue
            codepoint1 = match.group('codepoint1')
            codepoint2 = match.group('codepoint2')
            if not codepoint2:
                codepoint2 = codepoint1
            for i in range(int(codepoint1, 16),
                           int(codepoint2, 16) + 1):
                charmap_dictionary[i] = match.group('hexutf8')
        sys.stderr.write('No “CHARMAP” or no “END CHARMAP” found in %s\n'
                         %file_name)
        exit(1)

def check_charmap(original_file_name, new_file_name):
    '''Report differences in the CHARMAP section between the old and the
    new file
    '''
    print('************************************************************')
    print('Report on CHARMAP:')
    ocharmap = create_charmap_dictionary(original_file_name)
    ncharmap = create_charmap_dictionary(new_file_name)
    print('------------------------------------------------------------')
    print('Total removed characters in newly generated CHARMAP: %d'
          %len(set(ocharmap)-set(ncharmap)))
    if ARGS.show_missing_characters:
        for key in sorted(set(ocharmap)-set(ncharmap)):
            print('removed: {:s}     {:s} {:s}'.format(
                unicode_utils.ucs_symbol(key),
                ocharmap[key],
                unicode_utils.UNICODE_ATTRIBUTES[key]['name'] \
                if key in unicode_utils.UNICODE_ATTRIBUTES else 'None'))
    print('------------------------------------------------------------')
    changed_charmap = {}
    for key in set(ocharmap).intersection(set(ncharmap)):
        if ocharmap[key] != ncharmap[key]:
            changed_charmap[key] = (ocharmap[key], ncharmap[key])
    print('Total changed characters in newly generated CHARMAP: %d'
          %len(changed_charmap))
    if ARGS.show_changed_characters:
        for key in sorted(changed_charmap):
            print('changed: {:s}     {:s}->{:s} {:s}'.format(
                unicode_utils.ucs_symbol(key),
                changed_charmap[key][0],
                changed_charmap[key][1],
                unicode_utils.UNICODE_ATTRIBUTES[key]['name'] \
                if key in unicode_utils.UNICODE_ATTRIBUTES else 'None'))
    print('------------------------------------------------------------')
    print('Total added characters in newly generated CHARMAP: %d'
          %len(set(ncharmap)-set(ocharmap)))
    if ARGS.show_added_characters:
        for key in sorted(set(ncharmap)-set(ocharmap)):
            print('added: {:s}     {:s} {:s}'.format(
                unicode_utils.ucs_symbol(key),
                ncharmap[key],
                unicode_utils.UNICODE_ATTRIBUTES[key]['name'] \
                if key in unicode_utils.UNICODE_ATTRIBUTES else 'None'))

def create_width_dictionary(file_name):
    '''Create a dictionary for all code points found in the WIDTH
    section of a file
    '''
    with open(file_name, mode='r') as utf8_file:
        width_dictionary = {}
        for line in utf8_file:
            if line.startswith('WIDTH'):
                break
        for line in utf8_file:
            if line.startswith('END WIDTH'):
                return width_dictionary
            match = re.match(
                r'^<U(?P<codepoint1>[0-9A-F]{4,8})>'
                +r'(:?\.\.\.<U(?P<codepoint2>[0-9-A-F]{4,8})>)?'
                +r'\s+(?P<width>[02])',
                line)
            if not match:
                continue
            codepoint1 = match.group('codepoint1')
            codepoint2 = match.group('codepoint2')
            if not codepoint2:
                codepoint2 = codepoint1
            for i in range(int(codepoint1, 16),
                           int(codepoint2, 16) + 1):
                width_dictionary[i] = int(match.group('width'))
        sys.stderr.write('No “WIDTH” or no “END WIDTH” found in %s\n' %file)

def check_width(original_file_name, new_file_name):
    '''Report differences in the WIDTH section between the old and the new
    file
    '''
    print('************************************************************')
    print('Report on WIDTH:')
    owidth = create_width_dictionary(original_file_name)
    nwidth = create_width_dictionary(new_file_name)
    print('------------------------------------------------------------')
    print('Total removed characters in newly generated WIDTH: %d'
          %len(set(owidth)-set(nwidth)))
    print('(Characters not in WIDTH get width 1 by default, '
          + 'i.e. these have width 1 now.)')
    if ARGS.show_missing_characters:
        for key in sorted(set(owidth)-set(nwidth)):
            print('removed: {:s} '.format(unicode_utils.ucs_symbol(key))
                  + '{:d} : '.format(owidth[key])
                  + 'eaw={:s} '.format(
                      unicode_utils.EAST_ASIAN_WIDTHS[key]
                      if key in unicode_utils.EAST_ASIAN_WIDTHS else 'None')
                  + 'category={:2s} '.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['category']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None')
                  + 'bidi={:3s} '.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['bidi']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None')
                  + 'name={:s}'.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['name']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None'))
    print('------------------------------------------------------------')
    changed_width = {}
    for key in set(owidth).intersection(set(nwidth)):
        if owidth[key] != nwidth[key]:
            changed_width[key] = (owidth[key], nwidth[key])
    print('Total changed characters in newly generated WIDTH: %d'
          %len(changed_width))
    if ARGS.show_changed_characters:
        for key in sorted(changed_width):
            print('changed width: {:s} '.format(unicode_utils.ucs_symbol(key))
                  + '{:d}->{:d} : '.format(changed_width[key][0],
                                          changed_width[key][1])
                  + 'eaw={:s} '.format(
                      unicode_utils.EAST_ASIAN_WIDTHS[key]
                      if key in unicode_utils.EAST_ASIAN_WIDTHS else 'None')
                  + 'category={:2s} '.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['category']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None')
                  + 'bidi={:3s} '.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['bidi']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None')
                  + 'name={:s}'.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['name']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None'))
    print('------------------------------------------------------------')
    print('Total added characters in newly generated WIDTH: %d'
          %len(set(nwidth)-set(owidth)))
    print('(Characters not in WIDTH get width 1 by default, '
          + 'i.e. these had width 1 before.)')
    if ARGS.show_added_characters:
        for key in sorted(set(nwidth)-set(owidth)):
            print('added: {:s} '.format(unicode_utils.ucs_symbol(key))
                  + '{:d} : '.format(nwidth[key])
                  + 'eaw={:s} '.format(
                      unicode_utils.EAST_ASIAN_WIDTHS[key]
                      if key in unicode_utils.EAST_ASIAN_WIDTHS else 'None')
                  + 'category={:2s} '.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['category']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None')
                  + 'bidi={:3s} '.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['bidi']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None')
                  + 'name={:s}'.format(
                      unicode_utils.UNICODE_ATTRIBUTES[key]['name']
                      if key in unicode_utils.UNICODE_ATTRIBUTES else 'None'))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='''
        Compare the contents of LC_CTYPE in two files and check for errors.
        ''')
    PARSER.add_argument(
        '-o', '--old_utf8_file',
        nargs='?',
        required=True,
        type=str,
        help='The old UTF-8 file.')
    PARSER.add_argument(
        '-n', '--new_utf8_file',
        nargs='?',
        required=True,
        type=str,
        help='The new UTF-8 file.')
    PARSER.add_argument(
        '-u', '--unicode_data_file',
        nargs='?',
        type=str,
        help='The UnicodeData.txt file to read.')
    PARSER.add_argument(
        '-e', '--east_asian_width_file',
        nargs='?',
        type=str,
        help='The EastAsianWidth.txt file to read.')
    PARSER.add_argument(
        '-a', '--show_added_characters',
        action='store_true',
        help='Show characters which were added in detail.')
    PARSER.add_argument(
        '-m', '--show_missing_characters',
        action='store_true',
        help='Show characters which were removed in detail.')
    PARSER.add_argument(
        '-c', '--show_changed_characters',
        action='store_true',
        help='Show characters whose width was changed in detail.')
    ARGS = PARSER.parse_args()

    if ARGS.unicode_data_file:
        unicode_utils.fill_attributes(ARGS.unicode_data_file)
    if ARGS.east_asian_width_file:
        unicode_utils.fill_east_asian_widths(ARGS.east_asian_width_file)
    check_charmap(ARGS.old_utf8_file, ARGS.new_utf8_file)
    check_width(ARGS.old_utf8_file, ARGS.new_utf8_file)
