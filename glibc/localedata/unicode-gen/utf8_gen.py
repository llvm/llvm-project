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

'''glibc/localedata/charmaps/UTF-8 file generator script

This script generates a glibc/localedata/charmaps/UTF-8 file
from Unicode data.

Usage: python3 utf8_gen.py UnicodeData.txt EastAsianWidth.txt

It will output UTF-8 file
'''

import argparse
import sys
import re
import unicode_utils

# Auxiliary tables for Hangul syllable names, see the Unicode 3.0 book,
# sections 3.11 and 4.4.

JAMO_INITIAL_SHORT_NAME = (
    'G', 'GG', 'N', 'D', 'DD', 'R', 'M', 'B', 'BB', 'S', 'SS', '', 'J', 'JJ',
    'C', 'K', 'T', 'P', 'H'
)

JAMO_MEDIAL_SHORT_NAME = (
    'A', 'AE', 'YA', 'YAE', 'EO', 'E', 'YEO', 'YE', 'O', 'WA', 'WAE', 'OE',
    'YO', 'U', 'WEO', 'WE', 'WI', 'YU', 'EU', 'YI', 'I'
)

JAMO_FINAL_SHORT_NAME = (
    '', 'G', 'GG', 'GS', 'N', 'NI', 'NH', 'D', 'L', 'LG', 'LM', 'LB', 'LS',
    'LT', 'LP', 'LH', 'M', 'B', 'BS', 'S', 'SS', 'NG', 'J', 'C', 'K', 'T',
    'P', 'H'
)

def process_range(start, end, outfile, name):
    '''Writes a range of code points into the CHARMAP section of the
    output file

    '''
    if 'Hangul Syllable' in name:
        # from glibc/localedata/ChangeLog:
        #
        #  2000-09-24  Bruno Haible  <haible@clisp.cons.org>
        #  * charmaps/UTF-8: Expand <Hangul Syllable> and <Private Use> ranges,
        #  so they become printable and carry a width. Comment out surrogate
        #  ranges. Add a WIDTH table
        #
        # So we expand the Hangul Syllables here:
        for i in range(int(start, 16), int(end, 16)+1 ):
            index2, index3 = divmod(i - 0xaC00, 28)
            index1, index2 = divmod(index2, 21)
            hangul_syllable_name = 'HANGUL SYLLABLE ' \
                                   + JAMO_INITIAL_SHORT_NAME[index1] \
                                   + JAMO_MEDIAL_SHORT_NAME[index2] \
                                   + JAMO_FINAL_SHORT_NAME[index3]
            outfile.write('{:<11s} {:<12s} {:s}\n'.format(
                unicode_utils.ucs_symbol(i), convert_to_hex(i),
                hangul_syllable_name))
        return
    # UnicodeData.txt file has contains code point ranges like this:
    #
    # 3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
    # 4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
    #
    # The glibc UTF-8 file splits ranges like these into shorter
    # ranges of 64 code points each:
    #
    # <U3400>..<U343F>     /xe3/x90/x80         <CJK Ideograph Extension A>
    # …
    # <U4D80>..<U4DB5>     /xe4/xb6/x80         <CJK Ideograph Extension A>
    for i in range(int(start, 16), int(end, 16), 64 ):
        if i > (int(end, 16)-64):
            outfile.write('{:s}..{:s} {:<12s} {:s}\n'.format(
                    unicode_utils.ucs_symbol(i),
                    unicode_utils.ucs_symbol(int(end,16)),
                    convert_to_hex(i),
                    name))
            break
        outfile.write('{:s}..{:s} {:<12s} {:s}\n'.format(
                unicode_utils.ucs_symbol(i),
                unicode_utils.ucs_symbol(i+63),
                convert_to_hex(i),
                name))

def process_charmap(flines, outfile):
    '''This function takes an array which contains *all* lines of
    of UnicodeData.txt and write lines to outfile as used in the

    CHARMAP
    …
    END CHARMAP

    section of the UTF-8 file in glibc/localedata/charmaps/UTF-8.

    Samples for input lines:

    0010;<control>;Cc;0;BN;;;;;N;DATA LINK ESCAPE;;;;
    3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
    4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
    D800;<Non Private Use High Surrogate, First>;Cs;0;L;;;;;N;;;;;
    DB7F;<Non Private Use High Surrogate, Last>;Cs;0;L;;;;;N;;;;;
    100000;<Plane 16 Private Use, First>;Co;0;L;;;;;N;;;;;
    10FFFD;<Plane 16 Private Use, Last>;Co;0;L;;;;;N;;;;;

    Samples for output lines (Unicode-Value UTF-8-HEX Unicode-Char-Name):

    <U0010>     /x10 DATA LINK ESCAPE
    <U3400>..<U343F>     /xe3/x90/x80 <CJK Ideograph Extension A>
    %<UD800>     /xed/xa0/x80 <Non Private Use High Surrogate, First>
    %<UDB7F>     /xed/xad/xbf <Non Private Use High Surrogate, Last>
    <U0010FFC0>..<U0010FFFD>     /xf4/x8f/xbf/x80 <Plane 16 Private Use>

    '''
    fields_start = []
    for line in flines:
        fields = line.split(";")
         # Some characters have “<control>” as their name. We try to
         # use the “Unicode 1.0 Name” (10th field in
         # UnicodeData.txt) for them.
         #
         # The Characters U+0080, U+0081, U+0084 and U+0099 have
         # “<control>” as their name but do not even have aa
         # ”Unicode 1.0 Name”. We could write code to take their
         # alternate names from NameAliases.txt.
        if fields[1] == "<control>" and fields[10]:
            fields[1] = fields[10]
        # Handling code point ranges like:
        #
        # 3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
        # 4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
        if fields[1].endswith(', First>') and not 'Surrogate,' in fields[1]:
            fields_start = fields
            continue
        if fields[1].endswith(', Last>') and not 'Surrogate,' in fields[1]:
            process_range(fields_start[0], fields[0],
                          outfile, fields[1][:-7]+'>')
            fields_start = []
            continue
        fields_start = []
        if 'Surrogate,' in fields[1]:
            # Comment out the surrogates in the UTF-8 file.
            # One could of course skip them completely but
            # the original UTF-8 file in glibc had them as
            # comments, so we keep these comment lines.
            outfile.write('%')
        outfile.write('{:<11s} {:<12s} {:s}\n'.format(
                unicode_utils.ucs_symbol(int(fields[0], 16)),
                convert_to_hex(int(fields[0], 16)),
                fields[1]))

def convert_to_hex(code_point):
    '''Converts a code point to a hexadecimal UTF-8 representation
    like /x**/x**/x**.'''
    # Getting UTF8 of Unicode characters.
    # In Python3, .encode('UTF-8') does not work for
    # surrogates. Therefore, we use this conversion table
    surrogates = {
        0xD800: '/xed/xa0/x80',
        0xDB7F: '/xed/xad/xbf',
        0xDB80: '/xed/xae/x80',
        0xDBFF: '/xed/xaf/xbf',
        0xDC00: '/xed/xb0/x80',
        0xDFFF: '/xed/xbf/xbf',
    }
    if code_point in surrogates:
        return surrogates[code_point]
    return ''.join([
        '/x{:02x}'.format(c) for c in chr(code_point).encode('UTF-8')
    ])

def write_header_charmap(outfile):
    '''Write the header on top of the CHARMAP section to the output file'''
    outfile.write("<code_set_name> UTF-8\n")
    outfile.write("<comment_char> %\n")
    outfile.write("<escape_char> /\n")
    outfile.write("<mb_cur_min> 1\n")
    outfile.write("<mb_cur_max> 6\n\n")
    outfile.write("% CHARMAP generated using utf8_gen.py\n")
    outfile.write("% alias ISO-10646/UTF-8\n")
    outfile.write("CHARMAP\n")

def write_header_width(outfile, unicode_version):
    '''Writes the header on top of the WIDTH section to the output file'''
    outfile.write('% Character width according to Unicode '
                  + '{:s}.\n'.format(unicode_version))
    outfile.write('% - Default width is 1.\n')
    outfile.write('% - Double-width characters have width 2; generated from\n')
    outfile.write('%        "grep \'^[^;]*;[WF]\' EastAsianWidth.txt"\n')
    outfile.write('% - Non-spacing characters have width 0; '
                  + 'generated from PropList.txt or\n')
    outfile.write('%   "grep \'^[^;]*;[^;]*;[^;]*;[^;]*;NSM;\' '
                  + 'UnicodeData.txt"\n')
    outfile.write('% - Format control characters have width 0; '
                  + 'generated from\n')
    outfile.write("%   \"grep '^[^;]*;[^;]*;Cf;' UnicodeData.txt\"\n")
#   Not needed covered by Cf
#    outfile.write("% - Zero width characters have width 0; generated from\n")
#    outfile.write("%   \"grep '^[^;]*;ZERO WIDTH ' UnicodeData.txt\"\n")
    outfile.write("WIDTH\n")

def process_width(outfile, ulines, elines, plines):
    '''ulines are lines from UnicodeData.txt, elines are lines from
    EastAsianWidth.txt containing characters with width “W” or “F”,
    plines are lines from PropList.txt which contain characters
    with the property “Prepended_Concatenation_Mark”.

    '''
    width_dict = {}
    for line in elines:
        fields = line.split(";")
        if not '..' in fields[0]:
            code_points = (fields[0], fields[0])
        else:
            code_points = fields[0].split("..")
        for key in range(int(code_points[0], 16),
                         int(code_points[1], 16)+1):
            width_dict[key] = 2

    for line in ulines:
        fields = line.split(";")
        if fields[4] == "NSM" or fields[2] in ("Cf", "Me", "Mn"):
            width_dict[int(fields[0], 16)] = 0

    for line in plines:
        # Characters with the property “Prepended_Concatenation_Mark”
        # should have the width 1:
        fields = line.split(";")
        if not '..' in fields[0]:
            code_points = (fields[0], fields[0])
        else:
            code_points = fields[0].split("..")
        for key in range(int(code_points[0], 16),
                         int(code_points[1], 16)+1):
            del width_dict[key] # default width is 1

    # handle special cases for compatibility
    for key in list((0x00AD,)):
        # https://www.cs.tut.fi/~jkorpela/shy.html
        if key in width_dict:
            del width_dict[key] # default width is 1
    for key in list(range(0x1160, 0x1200)):
        # Hangul jungseong and jongseong:
        if key in unicode_utils.UNICODE_ATTRIBUTES:
            width_dict[key] = 0
    for key in list(range(0xD7B0, 0xD800)):
        # Hangul jungseong and jongseong:
        if key in unicode_utils.UNICODE_ATTRIBUTES:
            width_dict[key] = 0
    for key in list(range(0x3248, 0x3250)):
        # These are “A” which means we can decide whether to treat them
        # as “W” or “N” based on context:
        # http://www.unicode.org/mail-arch/unicode-ml/y2017-m08/0023.html
        # For us, “W” seems better.
        width_dict[key] = 2
    for key in list(range(0x4DC0, 0x4E00)):
        width_dict[key] = 2

    same_width_lists = []
    current_width_list = []
    for key in sorted(width_dict):
        if not current_width_list:
            current_width_list = [key]
        elif (key == current_width_list[-1] + 1
              and width_dict[key] == width_dict[current_width_list[0]]):
            current_width_list.append(key)
        else:
            same_width_lists.append(current_width_list)
            current_width_list = [key]
    if current_width_list:
        same_width_lists.append(current_width_list)

    for same_width_list in same_width_lists:
        if len(same_width_list) == 1:
            outfile.write('{:s}\t{:d}\n'.format(
                unicode_utils.ucs_symbol(same_width_list[0]),
                width_dict[same_width_list[0]]))
        else:
            outfile.write('{:s}...{:s}\t{:d}\n'.format(
                unicode_utils.ucs_symbol(same_width_list[0]),
                unicode_utils.ucs_symbol(same_width_list[-1]),
                width_dict[same_width_list[0]]))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='''
        Generate a UTF-8 file from UnicodeData.txt, EastAsianWidth.txt, and PropList.txt.
        ''')
    PARSER.add_argument(
        '-u', '--unicode_data_file',
        nargs='?',
        type=str,
        default='UnicodeData.txt',
        help=('The UnicodeData.txt file to read, '
              + 'default: %(default)s'))
    PARSER.add_argument(
        '-e', '--east_asian_with_file',
        nargs='?',
        type=str,
        default='EastAsianWidth.txt',
        help=('The EastAsianWidth.txt file to read, '
              + 'default: %(default)s'))
    PARSER.add_argument(
        '-p', '--prop_list_file',
        nargs='?',
        type=str,
        default='PropList.txt',
        help=('The PropList.txt file to read, '
              + 'default: %(default)s'))
    PARSER.add_argument(
        '--unicode_version',
        nargs='?',
        required=True,
        type=str,
        help='The Unicode version of the input files used.')
    ARGS = PARSER.parse_args()

    unicode_utils.fill_attributes(ARGS.unicode_data_file)
    with open(ARGS.unicode_data_file, mode='r') as UNIDATA_FILE:
        UNICODE_DATA_LINES = UNIDATA_FILE.readlines()
    with open(ARGS.east_asian_with_file, mode='r') as EAST_ASIAN_WIDTH_FILE:
        EAST_ASIAN_WIDTH_LINES = []
        for LINE in EAST_ASIAN_WIDTH_FILE:
            # If characters from EastAasianWidth.txt which are from
            # from reserved ranges (i.e. not yet assigned code points)
            # are added to the WIDTH section of the UTF-8 file, then
            # “make check” produces “Unknown Character” errors for
            # these code points because such unassigned code points
            # are not in the CHARMAP section of the UTF-8 file.
            #
            # Therefore, we skip all reserved code points when reading
            # the EastAsianWidth.txt file.
            if re.match(r'.*<reserved-.+>\.\.<reserved-.+>.*', LINE):
                continue
            if re.match(r'^[^;]*;[WF]', LINE):
                EAST_ASIAN_WIDTH_LINES.append(LINE.strip())
    with open(ARGS.prop_list_file, mode='r') as PROP_LIST_FILE:
        PROP_LIST_LINES = []
        for LINE in PROP_LIST_FILE:
            if re.match(r'^[^;]*;[\s]*Prepended_Concatenation_Mark', LINE):
                PROP_LIST_LINES.append(LINE.strip())
    with open('UTF-8', mode='w') as OUTFILE:
        # Processing UnicodeData.txt and write CHARMAP to UTF-8 file
        write_header_charmap(OUTFILE)
        process_charmap(UNICODE_DATA_LINES, OUTFILE)
        OUTFILE.write("END CHARMAP\n\n")
        # Processing EastAsianWidth.txt and write WIDTH to UTF-8 file
        write_header_width(OUTFILE, ARGS.unicode_version)
        process_width(OUTFILE,
                      UNICODE_DATA_LINES,
                      EAST_ASIAN_WIDTH_LINES,
                      PROP_LIST_LINES)
        OUTFILE.write("END WIDTH\n")
