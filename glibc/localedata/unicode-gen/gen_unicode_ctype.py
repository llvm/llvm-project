#!/usr/bin/python3
#
# Generate a Unicode conforming LC_CTYPE category from a UnicodeData file.
# Copyright (C) 2014-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Based on gen-unicode-ctype.c by Bruno Haible <haible@clisp.cons.org>, 2000.
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
Generate a Unicode conforming LC_CTYPE category from UnicodeData.txt and
DerivedCoreProperties.txt files.

To see how this script is used, call it with the “-h” option:

    $ ./gen_unicode_ctype.py -h
    … prints usage message …
'''

import argparse
import time
import re
import unicode_utils

def code_point_ranges(is_class_function):
    '''Returns a list of ranges of code points for which is_class_function
    returns True.

    Example:

    [[65, 90], [192, 214], [216, 222], [256], … ]
    '''
    cp_ranges  = []
    for code_point in sorted(unicode_utils.UNICODE_ATTRIBUTES):
        if is_class_function(code_point):
            if (cp_ranges
                and cp_ranges[-1][-1] == code_point - 1):
                if len(cp_ranges[-1]) == 1:
                    cp_ranges[-1].append(code_point)
                else:
                    cp_ranges[-1][-1] = code_point
            else:
                cp_ranges.append([code_point])
    return cp_ranges

def output_charclass(i18n_file, class_name, is_class_function):
    '''Output a LC_CTYPE character class section

    Example:

    upper /
       <U0041>..<U005A>;<U00C0>..<U00D6>;<U00D8>..<U00DE>;<U0100>;<U0102>;/
       …
       <U0001D790>..<U0001D7A8>;<U0001D7CA>;<U0001F130>..<U0001F149>;/
       <U0001F150>..<U0001F169>;<U0001F170>..<U0001F189>
    '''
    cp_ranges = code_point_ranges(is_class_function)
    if cp_ranges:
        i18n_file.write('%s /\n' %class_name)
        max_column = 75
        prefix = '   '
        line = prefix
        range_string = ''
        for code_point_range in cp_ranges:
            if line.strip():
                line  += ';'
            if len(code_point_range) == 1:
                range_string = unicode_utils.ucs_symbol(code_point_range[0])
            else:
                range_string = unicode_utils.ucs_symbol_range(
                    code_point_range[0], code_point_range[-1])
            if len(line+range_string) > max_column:
                i18n_file.write(line+'/\n')
                line = prefix
            line += range_string
        if line.strip():
            i18n_file.write(line+'\n')
        i18n_file.write('\n')

def output_charmap(i18n_file, map_name, map_function):
    '''Output a LC_CTYPE character map section

    Example:

    toupper /
      (<U0061>,<U0041>);(<U0062>,<U0042>);(<U0063>,<U0043>);(<U0064>,<U0044>);/
      …
      (<U000118DC>,<U000118BC>);(<U000118DD>,<U000118BD>);/
      (<U000118DE>,<U000118BE>);(<U000118DF>,<U000118BF>)
    '''
    max_column = 75
    prefix = '   '
    line = prefix
    map_string = ''
    i18n_file.write('%s /\n' %map_name)
    for code_point in sorted(unicode_utils.UNICODE_ATTRIBUTES):
        mapped = map_function(code_point)
        if code_point != mapped:
            if line.strip():
                line += ';'
            map_string = '(' \
                         + unicode_utils.ucs_symbol(code_point) \
                         + ',' \
                         + unicode_utils.ucs_symbol(mapped) \
                         + ')'
            if len(line+map_string) > max_column:
                i18n_file.write(line+'/\n')
                line = prefix
            line += map_string
    if line.strip():
        i18n_file.write(line+'\n')
    i18n_file.write('\n')

def read_input_file(filename):
    '''Reads the original glibc i18n file to get the original head
    and tail.

    We want to replace only the character classes in LC_CTYPE, and the
    date stamp. All the rest of the i18n file should stay unchanged.
    To avoid having to cut and paste the generated data into the
    original file, it is helpful to read the original file here
    to be able to generate a complete result file.
    '''
    head = tail = ''
    with open(filename, mode='r') as i18n_file:
        for line in i18n_file:
            match = re.match(
                r'^(?P<key>date\s+)(?P<value>"[0-9]{4}-[0-9]{2}-[0-9]{2}")',
                line)
            if match:
                line = match.group('key') \
                       + '"{:s}"\n'.format(time.strftime('%Y-%m-%d'))
            head = head + line
            if line.startswith('LC_CTYPE'):
                break
        for line in i18n_file:
            if line.startswith('translit_start'):
                tail = line
                break
        for line in i18n_file:
            tail = tail + line
    return (head, tail)

def output_head(i18n_file, unicode_version, head=''):
    '''Write the header of the output file, i.e. the part of the file
    before the “LC_CTYPE” line.
    '''
    if ARGS.input_file and head:
        i18n_file.write(head)
    else:
        i18n_file.write('escape_char /\n')
        i18n_file.write('comment_char %\n')
        i18n_file.write('\n')
        i18n_file.write('% Generated automatically by '
                        + 'gen_unicode_ctype.py '
                        + 'for Unicode {:s}.\n'.format(unicode_version))
        i18n_file.write('\n')
        i18n_file.write('LC_IDENTIFICATION\n')
        i18n_file.write('title     "Unicode {:s} FDCC-set"\n'.format(
            unicode_version))
        i18n_file.write('source    "UnicodeData.txt, '
                        + 'DerivedCoreProperties.txt"\n')
        i18n_file.write('address   ""\n')
        i18n_file.write('contact   ""\n')
        i18n_file.write('email     "bug-glibc-locales@gnu.org"\n')
        i18n_file.write('tel       ""\n')
        i18n_file.write('fax       ""\n')
        i18n_file.write('language  ""\n')
        i18n_file.write('territory "Earth"\n')
        i18n_file.write('revision  "{:s}"\n'.format(unicode_version))
        i18n_file.write('date      "{:s}"\n'.format(
            time.strftime('%Y-%m-%d')))
        i18n_file.write('category  "i18n:2012";LC_CTYPE\n')
        i18n_file.write('END LC_IDENTIFICATION\n')
        i18n_file.write('\n')
        i18n_file.write('LC_CTYPE\n')

def output_tail(i18n_file, tail=''):
    '''Write the tail of the output file, i.e. the part of the file
    after the last “LC_CTYPE” character class.
    '''
    if ARGS.input_file and tail:
        i18n_file.write(tail)
    else:
        i18n_file.write('END LC_CTYPE\n')

def output_tables(i18n_file, unicode_version, turkish):
    '''Write the new LC_CTYPE character classes to the output file'''
    i18n_file.write('% The following is the 14652 i18n fdcc-set '
                    + 'LC_CTYPE category.\n')
    i18n_file.write('% It covers Unicode version {:s}.\n'.format(
        unicode_version))
    i18n_file.write('% The character classes and mapping tables were '
                    + 'automatically\n')
    i18n_file.write('% generated using the gen_unicode_ctype.py '
                    + 'program.\n\n')
    i18n_file.write('% The "upper" class reflects the uppercase '
                    + 'characters of class "alpha"\n')
    output_charclass(i18n_file, 'upper', unicode_utils.is_upper)
    i18n_file.write('% The "lower" class reflects the lowercase '
                    + 'characters of class "alpha"\n')
    output_charclass(i18n_file, 'lower', unicode_utils.is_lower)
    i18n_file.write('% The "alpha" class of the "i18n" FDCC-set is '
                    + 'reflecting\n')
    i18n_file.write('% the recommendations in TR 10176 annex A\n')
    output_charclass(i18n_file, 'alpha', unicode_utils.is_alpha)
    i18n_file.write('% The "digit" class must only contain the '
                    + 'BASIC LATIN digits, says ISO C 99\n')
    i18n_file.write('% (sections 7.25.2.1.5 and 5.2.1).\n')
    output_charclass(i18n_file, 'digit', unicode_utils.is_digit)
    i18n_file.write('% The "outdigit" information is by default '
                    + '"0" to "9".  We don\'t have to\n')
    i18n_file.write('% provide it here since localedef will fill '
               + 'in the bits and it would\n')
    i18n_file.write('% prevent locales copying this file define '
                    + 'their own values.\n')
    i18n_file.write('% outdigit /\n')
    i18n_file.write('%    <U0030>..<U0039>\n\n')
    # output_charclass(i18n_file, 'outdigit', is_outdigit)
    output_charclass(i18n_file, 'space', unicode_utils.is_space)
    output_charclass(i18n_file, 'cntrl', unicode_utils.is_cntrl)
    output_charclass(i18n_file, 'punct', unicode_utils.is_punct)
    output_charclass(i18n_file, 'graph', unicode_utils.is_graph)
    output_charclass(i18n_file, 'print', unicode_utils.is_print)
    i18n_file.write('% The "xdigit" class must only contain the '
                    + 'BASIC LATIN digits and A-F, a-f,\n')
    i18n_file.write('% says ISO C 99 '
                    + '(sections 7.25.2.1.12 and 6.4.4.1).\n')
    output_charclass(i18n_file, 'xdigit', unicode_utils.is_xdigit)
    output_charclass(i18n_file, 'blank', unicode_utils.is_blank)
    if turkish:
        i18n_file.write('% The case conversions reflect '
                        + 'Turkish conventions.\n')
        output_charmap(i18n_file, 'toupper', unicode_utils.to_upper_turkish)
        output_charmap(i18n_file, 'tolower', unicode_utils.to_lower_turkish)
    else:
        output_charmap(i18n_file, 'toupper', unicode_utils.to_upper)
        output_charmap(i18n_file, 'tolower', unicode_utils.to_lower)
    output_charmap(i18n_file, 'map "totitle";', unicode_utils.to_title)
    i18n_file.write('% The "combining" class reflects ISO/IEC 10646-1 '
                    + 'annex B.1\n')
    i18n_file.write('% That is, all combining characters (level 2+3).\n')
    output_charclass(i18n_file, 'class "combining";',
                     unicode_utils.is_combining)
    i18n_file.write('% The "combining_level3" class reflects '
                    + 'ISO/IEC 10646-1 annex B.2\n')
    i18n_file.write('% That is, combining characters of level 3.\n')
    output_charclass(i18n_file, 'class "combining_level3";',
                     unicode_utils.is_combining_level3)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='''
        Generate a Unicode conforming LC_CTYPE category from
        UnicodeData.txt and DerivedCoreProperties.txt files.
        ''')
    PARSER.add_argument(
        '-u', '--unicode_data_file',
        nargs='?',
        type=str,
        default='UnicodeData.txt',
        help=('The UnicodeData.txt file to read, '
              + 'default: %(default)s'))
    PARSER.add_argument(
        '-d', '--derived_core_properties_file',
        nargs='?',
        type=str,
        default='DerivedCoreProperties.txt',
        help=('The DerivedCoreProperties.txt file to read, '
              + 'default: %(default)s'))
    PARSER.add_argument(
        '-i', '--input_file',
        nargs='?',
        type=str,
        help='''The original glibc/localedata/locales/i18n file.''')
    PARSER.add_argument(
        '-o', '--output_file',
        nargs='?',
        type=str,
        default='i18n.new',
        help='''The file which shall contain the generated LC_CTYPE category,
        default: %(default)s.  If the original
        glibc/localedata/locales/i18n has been given
        as an option, all data from the original file
        except the newly generated LC_CTYPE character
        classes and the date stamp in
        LC_IDENTIFICATION will be copied unchanged
        into the output file.  ''')
    PARSER.add_argument(
        '--unicode_version',
        nargs='?',
        required=True,
        type=str,
        help='The Unicode version of the input files used.')
    PARSER.add_argument(
        '--turkish',
        action='store_true',
        help='Use Turkish case conversions.')
    ARGS = PARSER.parse_args()

    unicode_utils.fill_attributes(
        ARGS.unicode_data_file)
    unicode_utils.fill_derived_core_properties(
        ARGS.derived_core_properties_file)
    unicode_utils.verifications()
    HEAD = TAIL = ''
    if ARGS.input_file:
        (HEAD, TAIL) = read_input_file(ARGS.input_file)
    with open(ARGS.output_file, mode='w') as I18N_FILE:
        output_head(I18N_FILE, ARGS.unicode_version, head=HEAD)
        output_tables(I18N_FILE, ARGS.unicode_version, ARGS.turkish)
        output_tail(I18N_FILE, tail=TAIL)
