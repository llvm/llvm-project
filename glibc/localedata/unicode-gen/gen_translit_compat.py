#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Generate a translit_compat file from a UnicodeData file.
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
Generate a translit_compat file from UnicodeData.txt

To see how this script is used, call it with the “-h” option:

    $ ./gen_translit_compat -h
    … prints usage message …
'''

import argparse
import time
import unicode_utils

def read_input_file(filename):
    '''Reads the original glibc translit_compat file to get the
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
        translit_file.write('% Transliterations of compatibility characters ')
        translit_file.write('and ligatures.\n')
        translit_file.write('% Generated automatically from UnicodeData.txt '
                            + 'by gen_translit_compat.py '
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

def compatibility_decompose(code_point):
    '''http://www.unicode.org/reports/tr44/#Character_Decomposition_Mappings

    “The compatibility decomposition is formed by recursively applying
    the canonical and compatibility mappings, then applying the
    Canonical Ordering Algorithm.”

    We don’t do the canonical decomposition here because this is
    done in gen_translit_combining.py to generate translit_combining.

    And we ignore some of the possible compatibility formatting tags
    here. Some of them are used in other translit_* files, not
    translit_compat:

    <font>:   translit_font
    <circle>: translit_circle
    <wide>:   translit_wide
    <narrow>: translit_narrow
    <square>: translit_cjk_compat
    <fraction>: translit_fraction

    And we ignore

    <noBreak>, <initial>, <medial>, <final>, <isolated>

    because they seem to be not useful for transliteration.
    '''
    decomposition = unicode_utils.UNICODE_ATTRIBUTES[
        code_point]['decomposition']
    compatibility_tags = (
        '<compat>', '<super>', '<sub>', '<vertical>')
    for compatibility_tag in compatibility_tags:
        if decomposition.startswith(compatibility_tag):
            decomposition = decomposition[len(compatibility_tag)+1:]
            decomposed_code_points = [int(x, 16)
                                      for x in decomposition.split(' ')]
            if (len(decomposed_code_points) > 1
                    and decomposed_code_points[0] == 0x0020
                    and decomposed_code_points[1] >= 0x0300
                    and decomposed_code_points[1] <= 0x03FF):
                # Decomposes into a space followed by a combining character.
                # This is not useful fo transliteration.
                return []
            else:
                return_value = []
                for index in range(0, len(decomposed_code_points)):
                    cd_code_points = compatibility_decompose(
                        decomposed_code_points[index])
                    if cd_code_points:
                        return_value += cd_code_points
                    else:
                        return_value += [decomposed_code_points[index]]
                return return_value
    return []

def special_decompose(code_point_list):
    '''
    Decompositions which are not in UnicodeData.txt at all but which
    were used in the original translit_compat file in glibc and
    which seem to make sense.  I want to keep the update of
    translit_compat close to the spirit of the original file,
    therefore I added this special decomposition rules here.
    '''
    special_decompose_dict = {
        (0x03BC,): [0x0075], # μ → u
        (0x02BC,): [0x0027], # ʼ → '
    }
    if tuple(code_point_list) in special_decompose_dict:
        return special_decompose_dict[tuple(code_point_list)]
    else:
        return code_point_list

def special_ligature_decompose(code_point):
    '''
    Decompositions for ligatures which are not in UnicodeData.txt at
    all but which were used in the original translit_compat file in
    glibc and which seem to make sense.  I want to keep the update of
    translit_compat close to the spirit of the original file,
    therefore I added these special ligature decomposition rules here.

    '''
    special_ligature_decompose_dict = {
        0x00E6: [0x0061, 0x0065], # æ → ae
        0x00C6: [0x0041, 0x0045], # Æ → AE
        # These following 5 special ligature decompositions were
        # in the original glibc/localedata/locales/translit_compat file
        0x0152: [0x004F, 0x0045], # Œ → OE
        0x0153: [0x006F, 0x0065], # œ → oe
        0x05F0: [0x05D5, 0x05D5], # װ → וו
        0x05F1: [0x05D5, 0x05D9], # ױ → וי
        0x05F2: [0x05D9, 0x05D9], # ײ → יי
        # The following special ligature decompositions were
        # not in the original glibc/localedata/locales/translit_compat file
        # U+04A4 CYRILLIC CAPITAL LIGATURE EN GHE
        # → U+041D CYRILLIC CAPITAL LETTER EN,
        #   U+0413 CYRILLIC CAPITAL LETTER GHE
        0x04A4: [0x041D, 0x0413], # Ҥ → НГ
        # U+04A5 CYRILLIC SMALL LIGATURE EN GHE
        # → U+043D CYRILLIC SMALL LETTER EN,
        #   U+0433 CYRILLIC SMALL LETTER GHE
        0x04A5: [0x043D, 0x0433], # ҥ → нг
        # U+04B4 CYRILLIC CAPITAL LIGATURE TE TSE
        # → U+0422 CYRILLIC CAPITAL LETTER TE,
        #   U+0426 CYRILLIC CAPITAL LETTER TSE
        0x04B4: [0x0422, 0x0426], # Ҵ → ТЦ
        # U+04B5 CYRILLIC SMALL LIGATURE TE TSE
        # → U+0442 CYRILLIC SMALL LETTER TE,
        #   U+0446 CYRILLIC SMALL LETTER TSE
        0x04B5: [0x0442, 0x0446], # ҵ → тц
        # U+04d4 CYRILLIC CAPITAL LIGATURE A IE
        # → U+0410 CYRILLIC CAPITAL LETTER A
        #   U+0415;CYRILLIC CAPITAL LETTER IE
        0x04D4: [0x0410, 0x0415], # Ӕ → АЕ
        # U+04D5 CYRILLIC SMALL LIGATURE A IE
        # → U+0430 CYRILLIC SMALL LETTER A,
        #   U+0435 CYRILLIC SMALL LETTER IE
        0x04D5: [0x0430, 0x0435], # ӕ → ае
        # I am not sure what to do with the following ligatures
        # maybe it makes no sense to decompose them:
        # U+0616 ARABIC SMALL HIGH LIGATURE ALEF WITH LAM WITH YEH
        # U+06d6 ARABIC SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA
        # U+06d7 ARABIC SMALL HIGH LIGATURE QAF WITH LAM WITH ALEF MAKSURA
        # U+fdfd ARABIC LIGATURE BISMILLAH AR-RAHMAN AR-RAHEEM
        # U+fe20 COMBINING LIGATURE LEFT HALF
        # U+fe21 COMBINING LIGATURE RIGHT HALF
        # U+fe27 COMBINING LIGATURE LEFT HALF BELOW
        # U+fe28 COMBINING LIGATURE RIGHT HALF BELOW
        # U+11176 MAHAJANI LIGATURE SHRI
        # U+1f670 SCRIPT LIGATURE ET ORNAMENT
        # U+1f671 HEAVY SCRIPT LIGATURE ET ORNAMENT
        # U+1f672 LIGATURE OPEN ET ORNAMENT
        # U+1f673 HEAVY LIGATURE OPEN ET ORNAMENT
    }
    if code_point in special_ligature_decompose_dict:
        return special_ligature_decompose_dict[code_point]
    else:
        return [code_point]

def output_transliteration(translit_file):
    '''Write the new transliteration to the output file'''
    translit_file.write('\n')
    for code_point in sorted(unicode_utils.UNICODE_ATTRIBUTES):
        name = unicode_utils.UNICODE_ATTRIBUTES[code_point]['name']
        decomposed_code_points = [compatibility_decompose(code_point)]
        if not decomposed_code_points[0]:
            if special_decompose([code_point]) != [code_point]:
                decomposed_code_points[0] = special_decompose([code_point])
        else:
            special_decomposed_code_points = []
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
        if decomposed_code_points[0]:
            translit_file.write('% {:s}\n'.format(name))
            translit_file.write('{:s} '.format(
                unicode_utils.ucs_symbol(code_point)))
            for index in range(0, len(decomposed_code_points)):
                if index > 0:
                    translit_file.write(';')
                translit_file.write('"')
                for decomposed_code_point in decomposed_code_points[index]:
                    translit_file.write('{:s}'.format(
                        unicode_utils.ucs_symbol(decomposed_code_point)))
                translit_file.write('"')
            translit_file.write('\n')
        elif 'LIGATURE' in name and 'ARABIC' not in name:
            decomposed_code_points = special_ligature_decompose(code_point)
            if decomposed_code_points[0] != code_point:
                translit_file.write('% {:s}\n'.format(name))
                translit_file.write('{:s} '.format(
                    unicode_utils.ucs_symbol(code_point)))
                translit_file.write('"')
                for decomposed_code_point in decomposed_code_points:
                    translit_file.write('{:s}'.format(
                        unicode_utils.ucs_symbol(decomposed_code_point)))
                translit_file.write('"')
                translit_file.write('\n')
            else:
                print('Warning: unhandled ligature: {:x} {:s}'.format(
                    code_point, name))
    translit_file.write('\n')

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='''
        Generate a translit_compat file from UnicodeData.txt.
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
        help=''' The original glibc/localedata/locales/translit_compat
        file.''')
    PARSER.add_argument(
        '-o', '--output_file',
        nargs='?',
        type=str,
        default='translit_compat.new',
        help='''The new translit_compat file, default: %(default)s.  If the
        original glibc/localedata/locales/translit_compat file has
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
