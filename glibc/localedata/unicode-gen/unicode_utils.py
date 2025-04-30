# Utilities to generate Unicode data for glibc from upstream Unicode data.
#
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

'''
This module contains utilities used by the scripts to generate
Unicode data for glibc from upstream Unicode data files.
'''

import sys
import re


# Common locale header.
COMMENT_HEADER = """
% This file is part of the GNU C Library and contains locale data.
% The Free Software Foundation does not claim any copyright interest
% in the locale data contained in this file.  The foregoing does not
% affect the license of the GNU C Library as a whole.  It does not
% exempt you from the conditions of the license if your use would
% otherwise be governed by that license.
"""

# Dictionary holding the entire contents of the UnicodeData.txt file
#
# Contents of this dictionary look like this:
#
# {0: {'category': 'Cc',
#      'title': None,
#      'digit': '',
#      'name': '<control>',
#      'bidi': 'BN',
#      'combining': '0',
#      'comment': '',
#      'oldname': 'NULL',
#      'decomposition': '',
#      'upper': None,
#      'mirrored': 'N',
#      'lower': None,
#      'decdigit': '',
#      'numeric': ''},
#      …
# }
UNICODE_ATTRIBUTES = {}

# Dictionary holding the entire contents of the DerivedCoreProperties.txt file
#
# Contents of this dictionary look like this:
#
# {917504: ['Default_Ignorable_Code_Point'],
#  917505: ['Case_Ignorable', 'Default_Ignorable_Code_Point'],
#  …
# }
DERIVED_CORE_PROPERTIES = {}

# Dictionary holding the entire contents of the EastAsianWidths.txt file
#
# Contents of this dictionary look like this:
#
# {0: 'N', … , 45430: 'W', …}
EAST_ASIAN_WIDTHS = {}

def fill_attribute(code_point, fields):
    '''Stores in UNICODE_ATTRIBUTES[code_point] the values from the fields.

    One entry in the UNICODE_ATTRIBUTES dictionary represents one line
    in the UnicodeData.txt file.

    '''
    UNICODE_ATTRIBUTES[code_point] =  {
        'name': fields[1],          # Character name
        'category': fields[2],      # General category
        'combining': fields[3],     # Canonical combining classes
        'bidi': fields[4],          # Bidirectional category
        'decomposition': fields[5], # Character decomposition mapping
        'decdigit': fields[6],      # Decimal digit value
        'digit': fields[7],         # Digit value
        'numeric': fields[8],       # Numeric value
        'mirrored': fields[9],      # mirrored
        'oldname': fields[10],      # Old Unicode 1.0 name
        'comment': fields[11],      # comment
        # Uppercase mapping
        'upper': int(fields[12], 16) if fields[12] else None,
        # Lowercase mapping
        'lower': int(fields[13], 16) if fields[13] else None,
        # Titlecase mapping
        'title': int(fields[14], 16) if fields[14] else None,
    }

def fill_attributes(filename):
    '''Stores the entire contents of the UnicodeData.txt file
    in the UNICODE_ATTRIBUTES dictionary.

    A typical line for a single code point in UnicodeData.txt looks
    like this:

    0041;LATIN CAPITAL LETTER A;Lu;0;L;;;;;N;;;;0061;

    Code point ranges are indicated by pairs of lines like this:

    4E00;<CJK Ideograph, First>;Lo;0;L;;;;;N;;;;;
    9FCC;<CJK Ideograph, Last>;Lo;0;L;;;;;N;;;;;
    '''
    with open(filename, mode='r') as unicode_data_file:
        fields_start = []
        for line in unicode_data_file:
            fields = line.strip().split(';')
            if len(fields) != 15:
                sys.stderr.write(
                    'short line in file "%(f)s": %(l)s\n' %{
                    'f': filename, 'l': line})
                exit(1)
            if fields[2] == 'Cs':
                # Surrogates are UTF-16 artefacts,
                # not real characters. Ignore them.
                fields_start = []
                continue
            if fields[1].endswith(', First>'):
                fields_start = fields
                fields_start[1] = fields_start[1].split(',')[0][1:]
                continue
            if fields[1].endswith(', Last>'):
                fields[1] = fields[1].split(',')[0][1:]
                if fields[1:] != fields_start[1:]:
                    sys.stderr.write(
                        'broken code point range in file "%(f)s": %(l)s\n' %{
                            'f': filename, 'l': line})
                    exit(1)
                for code_point in range(
                        int(fields_start[0], 16),
                        int(fields[0], 16)+1):
                    fill_attribute(code_point, fields)
                fields_start = []
                continue
            fill_attribute(int(fields[0], 16), fields)
            fields_start = []

def fill_derived_core_properties(filename):
    '''Stores the entire contents of the DerivedCoreProperties.txt file
    in the DERIVED_CORE_PROPERTIES dictionary.

    Lines in DerivedCoreProperties.txt are either a code point range like
    this:

    0061..007A    ; Lowercase # L&  [26] LATIN SMALL LETTER A..LATIN SMALL LETTER Z

    or a single code point like this:

    00AA          ; Lowercase # Lo       FEMININE ORDINAL INDICATOR

    '''
    with open(filename, mode='r') as derived_core_properties_file:
        for line in derived_core_properties_file:
            match = re.match(
                r'^(?P<codepoint1>[0-9A-F]{4,6})'
                + r'(?:\.\.(?P<codepoint2>[0-9A-F]{4,6}))?'
                + r'\s*;\s*(?P<property>[a-zA-Z_]+)',
                line)
            if not match:
                continue
            start = match.group('codepoint1')
            end = match.group('codepoint2')
            if not end:
                end = start
            for code_point in range(int(start, 16), int(end, 16)+1):
                prop = match.group('property')
                if code_point in DERIVED_CORE_PROPERTIES:
                    DERIVED_CORE_PROPERTIES[code_point].append(prop)
                else:
                    DERIVED_CORE_PROPERTIES[code_point] = [prop]

def fill_east_asian_widths(filename):
    '''Stores the entire contents of the EastAsianWidths.txt file
    in the EAST_ASIAN_WIDTHS dictionary.

    Lines in EastAsianWidths.txt are either a code point range like
    this:

    9FCD..9FFF;W     # Cn    [51] <reserved-9FCD>..<reserved-9FFF>

    or a single code point like this:

    A015;W           # Lm         YI SYLLABLE WU
    '''
    with open(filename, mode='r') as east_asian_widths_file:
        for line in east_asian_widths_file:
            match = re.match(
                r'^(?P<codepoint1>[0-9A-F]{4,6})'
                +r'(?:\.\.(?P<codepoint2>[0-9A-F]{4,6}))?'
                +r'\s*;\s*(?P<property>[a-zA-Z]+)',
                line)
            if not match:
                continue
            start = match.group('codepoint1')
            end = match.group('codepoint2')
            if not end:
                end = start
            for code_point in range(int(start, 16), int(end, 16)+1):
                EAST_ASIAN_WIDTHS[code_point] = match.group('property')

def to_upper(code_point):
    '''Returns the code point of the uppercase version
    of the given code point'''
    if (UNICODE_ATTRIBUTES[code_point]['name']
        and UNICODE_ATTRIBUTES[code_point]['upper']):
        return UNICODE_ATTRIBUTES[code_point]['upper']
    else:
        return code_point

def to_lower(code_point):
    '''Returns the code point of the lowercase version
    of the given code point'''
    if (UNICODE_ATTRIBUTES[code_point]['name']
        and UNICODE_ATTRIBUTES[code_point]['lower']):
        return UNICODE_ATTRIBUTES[code_point]['lower']
    else:
        return code_point

def to_upper_turkish(code_point):
    '''Returns the code point of the Turkish uppercase version
    of the given code point'''
    if code_point == 0x0069:
        return 0x0130
    return to_upper(code_point)

def to_lower_turkish(code_point):
    '''Returns the code point of the Turkish lowercase version
    of the given code point'''
    if code_point == 0x0049:
        return 0x0131
    return to_lower(code_point)

def to_title(code_point):
    '''Returns the code point of the titlecase version
    of the given code point'''
    if (UNICODE_ATTRIBUTES[code_point]['name']
        and UNICODE_ATTRIBUTES[code_point]['title']):
        return UNICODE_ATTRIBUTES[code_point]['title']
    else:
        return code_point

def is_upper(code_point):
    '''Checks whether the character with this code point is uppercase'''
    return (to_lower(code_point) != code_point
            or (code_point in DERIVED_CORE_PROPERTIES
                and 'Uppercase' in DERIVED_CORE_PROPERTIES[code_point]))

def is_lower(code_point):
    '''Checks whether the character with this code point is lowercase'''
    # Some characters are defined as “Lowercase” in
    # DerivedCoreProperties.txt but do not have a mapping to upper
    # case. For example, ꜰ U+A72F “LATIN LETTER SMALL CAPITAL F” is
    # one of these.
    return (to_upper(code_point) != code_point
            # <U00DF> is lowercase, but without simple to_upper mapping.
            or code_point == 0x00DF
            or (code_point in DERIVED_CORE_PROPERTIES
                and 'Lowercase' in DERIVED_CORE_PROPERTIES[code_point]))

def is_alpha(code_point):
    '''Checks whether the character with this code point is alphabetic'''
    return ((code_point in DERIVED_CORE_PROPERTIES
             and
             'Alphabetic' in DERIVED_CORE_PROPERTIES[code_point])
            or
            # Consider all the non-ASCII digits as alphabetic.
            # ISO C 99 forbids us to have them in category “digit”,
            # but we want iswalnum to return true on them.
            (UNICODE_ATTRIBUTES[code_point]['category'] == 'Nd'
             and not (code_point >= 0x0030 and code_point <= 0x0039)))

def is_digit(code_point):
    '''Checks whether the character with this code point is a digit'''
    if False:
        return (UNICODE_ATTRIBUTES[code_point]['name']
                and UNICODE_ATTRIBUTES[code_point]['category'] == 'Nd')
        # Note: U+0BE7..U+0BEF and U+1369..U+1371 are digit systems without
        # a zero.  Must add <0> in front of them by hand.
    else:
        # SUSV2 gives us some freedom for the "digit" category, but ISO C 99
        # takes it away:
        # 7.25.2.1.5:
        #    The iswdigit function tests for any wide character that
        #    corresponds to a decimal-digit character (as defined in 5.2.1).
        # 5.2.1:
        #    the 10 decimal digits 0 1 2 3 4 5 6 7 8 9
        return (code_point >= 0x0030 and code_point <= 0x0039)

def is_outdigit(code_point):
    '''Checks whether the character with this code point is outdigit'''
    return (code_point >= 0x0030 and code_point <= 0x0039)

def is_blank(code_point):
    '''Checks whether the character with this code point is blank'''
    return (code_point == 0x0009 # '\t'
            # Category Zs without mention of '<noBreak>'
            or (UNICODE_ATTRIBUTES[code_point]['name']
                and UNICODE_ATTRIBUTES[code_point]['category'] == 'Zs'
                and '<noBreak>' not in
                UNICODE_ATTRIBUTES[code_point]['decomposition']))

def is_space(code_point):
    '''Checks whether the character with this code point is a space'''
    # Don’t make U+00A0 a space. Non-breaking space means that all programs
    # should treat it like a punctuation character, not like a space.
    return (code_point == 0x0020 # ' '
            or code_point == 0x000C # '\f'
            or code_point == 0x000A # '\n'
            or code_point == 0x000D # '\r'
            or code_point == 0x0009 # '\t'
            or code_point == 0x000B # '\v'
            # Categories Zl, Zp, and Zs without mention of "<noBreak>"
            or (UNICODE_ATTRIBUTES[code_point]['name']
                and
                (UNICODE_ATTRIBUTES[code_point]['category'] in ['Zl', 'Zp']
                 or
                 (UNICODE_ATTRIBUTES[code_point]['category'] in ['Zs']
                  and
                  '<noBreak>' not in
                  UNICODE_ATTRIBUTES[code_point]['decomposition']))))

def is_cntrl(code_point):
    '''Checks whether the character with this code point is
    a control character'''
    return (UNICODE_ATTRIBUTES[code_point]['name']
            and (UNICODE_ATTRIBUTES[code_point]['name'] == '<control>'
                 or
                 UNICODE_ATTRIBUTES[code_point]['category'] in ['Zl', 'Zp']))

def is_xdigit(code_point):
    '''Checks whether the character with this code point is
    a hexadecimal digit'''
    if False:
        return (is_digit(code_point)
                or (code_point >= 0x0041 and code_point <= 0x0046)
                or (code_point >= 0x0061 and code_point <= 0x0066))
    else:
        # SUSV2 gives us some freedom for the "xdigit" category, but ISO C 99
        # takes it away:
        # 7.25.2.1.12:
        #    The iswxdigit function tests for any wide character that
        #    corresponds to a hexadecimal-digit character (as defined
        #    in 6.4.4.1).
        # 6.4.4.1:
        #    hexadecimal-digit: one of
        #    0 1 2 3 4 5 6 7 8 9 a b c d e f A B C D E F
        return ((code_point >= 0x0030 and code_point  <= 0x0039)
                or (code_point >= 0x0041 and code_point <= 0x0046)
                or (code_point >= 0x0061 and code_point <= 0x0066))

def is_graph(code_point):
    '''Checks whether the character with this code point is
    a graphical character'''
    return (UNICODE_ATTRIBUTES[code_point]['name']
            and UNICODE_ATTRIBUTES[code_point]['name'] != '<control>'
            and not is_space(code_point))

def is_print(code_point):
    '''Checks whether the character with this code point is printable'''
    return (UNICODE_ATTRIBUTES[code_point]['name']
            and UNICODE_ATTRIBUTES[code_point]['name'] != '<control>'
            and UNICODE_ATTRIBUTES[code_point]['category'] not in ['Zl', 'Zp'])

def is_punct(code_point):
    '''Checks whether the character with this code point is punctuation'''
    if False:
        return (UNICODE_ATTRIBUTES[code_point]['name']
                and UNICODE_ATTRIBUTES[code_point]['category'].startswith('P'))
    else:
        # The traditional POSIX definition of punctuation is every graphic,
        # non-alphanumeric character.
        return (is_graph(code_point)
                and not is_alpha(code_point)
                and not is_digit(code_point))

def is_combining(code_point):
    '''Checks whether the character with this code point is
    a combining character'''
    # Up to Unicode 3.0.1 we took the Combining property from the PropList.txt
    # file. In 3.0.1 it was identical to the union of the general categories
    # "Mn", "Mc", "Me". In Unicode 3.1 this property has been dropped from the
    # PropList.txt file, so we take the latter definition.
    return (UNICODE_ATTRIBUTES[code_point]['name']
            and
            UNICODE_ATTRIBUTES[code_point]['category'] in ['Mn', 'Mc', 'Me'])

def is_combining_level3(code_point):
    '''Checks whether the character with this code point is
    a combining level3 character'''
    return (is_combining(code_point)
            and
            int(UNICODE_ATTRIBUTES[code_point]['combining']) in range(0, 200))

def ucs_symbol(code_point):
    '''Return the UCS symbol string for a Unicode character.'''
    if code_point < 0x10000:
        return '<U{:04X}>'.format(code_point)
    else:
        return '<U{:08X}>'.format(code_point)

def ucs_symbol_range(code_point_low, code_point_high):
    '''Returns a string UCS symbol string for a code point range.

    Example:

    <U0041>..<U005A>
    '''
    return ucs_symbol(code_point_low) + '..' + ucs_symbol(code_point_high)

def verifications():
    '''Tests whether the is_* functions observe the known restrictions'''
    for code_point in sorted(UNICODE_ATTRIBUTES):
        # toupper restriction: "Only characters specified for the keywords
        # lower and upper shall be specified.
        if (to_upper(code_point) != code_point
            and not (is_lower(code_point) or is_upper(code_point))):
            sys.stderr.write(
                ('%(sym)s is not upper|lower '
                 + 'but toupper(0x%(c)04X) = 0x%(uc)04X\n') %{
                    'sym': ucs_symbol(code_point),
                    'c': code_point,
                    'uc': to_upper(code_point)})
        # tolower restriction: "Only characters specified for the keywords
        # lower and upper shall be specified.
        if (to_lower(code_point) != code_point
            and not (is_lower(code_point) or is_upper(code_point))):
            sys.stderr.write(
                ('%(sym)s is not upper|lower '
                 + 'but tolower(0x%(c)04X) = 0x%(uc)04X\n') %{
                    'sym': ucs_symbol(code_point),
                    'c': code_point,
                    'uc': to_lower(code_point)})
        # alpha restriction: "Characters classified as either upper or lower
        # shall automatically belong to this class.
        if ((is_lower(code_point) or is_upper(code_point))
             and not is_alpha(code_point)):
            sys.stderr.write('%(sym)s is upper|lower but not alpha\n' %{
                'sym': ucs_symbol(code_point)})
        # alpha restriction: “No character specified for the keywords cntrl,
        # digit, punct or space shall be specified.”
        if (is_alpha(code_point) and is_cntrl(code_point)):
            sys.stderr.write('%(sym)s is alpha and cntrl\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_alpha(code_point) and is_digit(code_point)):
            sys.stderr.write('%(sym)s is alpha and digit\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_alpha(code_point) and is_punct(code_point)):
            sys.stderr.write('%(sym)s is alpha and punct\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_alpha(code_point) and is_space(code_point)):
            sys.stderr.write('%(sym)s is alpha and space\n' %{
                'sym': ucs_symbol(code_point)})
        # space restriction: “No character specified for the keywords upper,
        # lower, alpha, digit, graph or xdigit shall be specified.”
        # upper, lower, alpha already checked above.
        if (is_space(code_point) and is_digit(code_point)):
            sys.stderr.write('%(sym)s is space and digit\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_space(code_point) and is_graph(code_point)):
            sys.stderr.write('%(sym)s is space and graph\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_space(code_point) and is_xdigit(code_point)):
            sys.stderr.write('%(sym)s is space and xdigit\n' %{
                'sym': ucs_symbol(code_point)})
        # cntrl restriction: “No character specified for the keywords upper,
        # lower, alpha, digit, punct, graph, print or xdigit shall be
        # specified.”  upper, lower, alpha already checked above.
        if (is_cntrl(code_point) and is_digit(code_point)):
            sys.stderr.write('%(sym)s is cntrl and digit\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_cntrl(code_point) and is_punct(code_point)):
            sys.stderr.write('%(sym)s is cntrl and punct\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_cntrl(code_point) and is_graph(code_point)):
            sys.stderr.write('%(sym)s is cntrl and graph\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_cntrl(code_point) and is_print(code_point)):
            sys.stderr.write('%(sym)s is cntrl and print\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_cntrl(code_point) and is_xdigit(code_point)):
            sys.stderr.write('%(sym)s is cntrl and xdigit\n' %{
                'sym': ucs_symbol(code_point)})
        # punct restriction: “No character specified for the keywords upper,
        # lower, alpha, digit, cntrl, xdigit or as the <space> character shall
        # be specified.”  upper, lower, alpha, cntrl already checked above.
        if (is_punct(code_point) and is_digit(code_point)):
            sys.stderr.write('%(sym)s is punct and digit\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_punct(code_point) and is_xdigit(code_point)):
            sys.stderr.write('%(sym)s is punct and xdigit\n' %{
                'sym': ucs_symbol(code_point)})
        if (is_punct(code_point) and code_point == 0x0020):
            sys.stderr.write('%(sym)s is punct\n' %{
                'sym': ucs_symbol(code_point)})
        # graph restriction: “No character specified for the keyword cntrl
        # shall be specified.”  Already checked above.

        # print restriction: “No character specified for the keyword cntrl
        # shall be specified.”  Already checked above.

        # graph - print relation: differ only in the <space> character.
        # How is this possible if there are more than one space character?!
        # I think susv2/xbd/locale.html should speak of “space characters”,
        # not “space character”.
        if (is_print(code_point)
            and not (is_graph(code_point) or is_space(code_point))):
            sys.stderr.write('%(sym)s is print but not graph|<space>\n' %{
                'sym': unicode_utils.ucs_symbol(code_point)})
        if (not is_print(code_point)
            and (is_graph(code_point) or code_point == 0x0020)):
            sys.stderr.write('%(sym)s is graph|<space> but not print\n' %{
                'sym': unicode_utils.ucs_symbol(code_point)})
