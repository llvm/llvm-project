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
This file contains a list of test cases used by
the ctype_compatibility.py script.
'''

TEST_CASES = [
    [[0x0E2F, 0x0E46], [('alpha', True), ('punct', False)],
     '''Theppitak Karoonboonyanan <thep@links.nectec.or.th> says
     <U0E2F>, <U0E46> should belong to punct. DerivedCoreProperties.txt
     says it is alpha. We trust DerivedCoreProperties.txt.'''
    ],
    [[0x0E31, (0x0E34, 0x0E3A)], [('alpha', True)],
     '''gen-unicode-ctype.c: Theppitak Karoonboonyanan
     <thep@links.nectec.or.th> says <U0E31>, <U0E34>..<U0E3A>
     are alpha. DerivedCoreProperties.txt agrees.'''
    ],
    [[(0x0E47, 0x0E4C), 0x0E4E], [('alpha', False)],
     '''gen-unicode-ctype.c: Theppitak Karoonboonyanan
     <thep@links.nectec.or.th> says <U0E47>..<U0E4E> are
     is_alpha. DerivedCoreProperties does says *only* <U0E4D>
     in that range is alphabetic, the others are *not*. We
     trust DerivedCoreProperties.txt.'''
    ],
    [[0x0E4D], [('alpha', True)],
     '''gen-unicode-ctype.c: Theppitak Karoonboonyanan
     <thep@links.nectec.or.th> says <U0E47>..<U0E4E> are
     is_alpha. DerivedCoreProperties does says *only* <U0E4D>
            in that range is alphabetic, the others are *not*. We
            trust DerivedCoreProperties.txt.
            '''
    ],
    [[0x0345], [('alpha', True), ('lower', True)],
     '''COMBINING GREEK YPOGEGRAMMENI
     According to DerivedCoreProperties.txt, this is “Alphabetic”
     and “Lowercase”.'''
    ],
    [[(0x2160, 0x2188)], [('alpha', True)],
     '''Roman Numerals are “Alphabetic” according to
     DerivedCoreProperties.txt'''
    ],
    [[(0x24B6, 0x24E9)], [('alpha', True)],
     '''Circled Latin letters are “Alphabetic” according to
     DerivedCoreProperties.txt'''
    ],
    [[0x661], [('alpha', True), ('digit', False)],
     '''gen-unicode-ctype.c: All non-ASCII digits should be alphabetic.
     ISO C 99 forbids us to have them in category "digit", but we
     want iswalnum to return true on them. Don’t forget to
     have a look at all the other digits, 0x661 is just one
     example tested here.'''
    ],
    [[(0x0030, 0x0039)], [('digit', True)],
     '''gen-unicode-ctype.c: All ASCII digits should be digits.'''
    ],
    [[0x0009], [('blank', True)],
     '''gen-unicode-ctype.c: CHARACTER TABULATION'''
    ],
    [[0x2007], [('blank', False), ('space', False)],
     '''gen-unicode-ctype.c: FIGURE SPACE, because it has <noBreak>
     in the description.'''
    ],
    [[0x0009, 0x000A, 0x000B, 0x000C, 0x000D], [('space', True)],
     '''gen-unicode-ctype.c: CHARACTER TABULATION, LINE FEED (LF), LINE
     TABULATION, ;FORM FEED (FF), CARRIAGE RETURN (CR)'''
    ],
    [[0x2028, 0x2029], [('cntrl', True)],
     '''gen-unicode-ctype.c: LINE SEPARATOR and PARAGRAPH SEPARATOR
     should be cntrl.'''
    ],
    [[(0x0030, 0x0039), (0x0041, 0x0046), (0x0061, 0x0066)],
     [('xdigit', True)],
     '''gen-unicode-ctype.c: ISO C 99 says (6.4.4.1): hexadecimal-digit:
     one of 0 1 2 3 4 5 6 7 8 9 a b c d e f A B C D E F (nothing else
     should be considered as a hexadecimal-digit)'''
    ],
    [[0x0330], [('combining', True), ('combining_level3', False)],
     '''gen-unicode-ctype.c: COMBINING TILDE BELOW, canonical combining
     class value >= 200, should be in combining but not in
     combining_level3'''
    ],
    [[0x0250, 0x0251, 0x0271], [('lower', True)],
     '''Should be lower in Unicode 7.0.0 (was not lower in
     Unicode 5.0.0).
     '''
    ],
    [[0x2184], [('lower', True)],
     '''Should be lower both in Unicode 5.0.0 and 7.0.0'''
    ],
    [[0xA67F], [('punct', False), ('alpha', True)],
     '''0xa67f CYRILLIC PAYEROK. Not in Unicode 5.0.0. In Unicode
     7.0.0. General category Lm (Letter
     modifier). DerivedCoreProperties.txt says it is
     “Alphabetic”. Apparently added manually to punct by mistake in
     glibc’s old LC_CTYPE.'''
    ],
    [[0xA60C], [('punct', False), ('alpha', True)],
     '''0xa60c VAI SYLLABLE LENGTHENER. Not in Unicode 5.0.0.
     In Unicode 7.0.0. General category Lm (Letter
     modifier). DerivedCoreProperties.txt says it is
     “Alphabetic”. Apparently added manually to punct by mistake in
     glibc’s old LC_CTYPE.'''
    ],
    [[0x2E2F], [('punct', False), ('alpha', True)],
     '''0x2E2F VERTICAL TILDE. Not in Unicode 5.0.0. In Unicode
     7.0.0. General category Lm (Letter
     modifier). DerivedCoreProperties.txt says it is
     “Alphabetic”. Apparently added manually to punct by mistake in
     glibc’s old LC_CTYPE.'''
    ],
    [[(0x1090, 0x1099)], [('punct', False), ('alpha', True)],
     '''MYANMAR SHAN DIGIT ZERO - MYANMAR SHAN DIGIT NINE.
     These are digits, but because ISO C 99 forbids to
     put them into digit they should go into alpha.'''
    ],
    [[0x103F], [('punct', False), ('alpha', True)],
     '''0x103F MYANMAR LETTER GREAT SA. Not in Unicode 5.0.0.
     In Unicode 7.0.0. General category Lo
     (Other_Letter). DerivedCoreProperties.txt says it is
     “Alphabetic”. Apparently added manually to punct by
     mistake in glibc’s old LC_CTYPE.'''
    ],
    [[0x0374], [('punct', False), ('alpha', True)],
     '''0x0374 GREEK NUMERAL SIGN. Unicode 5.0.0: general category
     Sk. Unicode 7.0.0: General category Lm
     (Modifier_Letter). DerivedCoreProperties.txt says it is
     “Alphabetic”.'''
    ],
    [[0x02EC], [('punct', False), ('alpha', True)],
     '''0x02EC MODIFIER LETTER VOICING. Unicode 5.0.0: general category
     Sk. Unicode 7.0.0: General category Lm
     (Modifier_Letter). DerivedCoreProperties.txt says it is
     “Alphabetic”.'''
    ],
    [[0x180E], [('space', False), ('blank', False)],
     '''0x180e MONGOLIAN VOWEL SEPARATOR. Unicode 5.0.0: General
     category Zs (Space_Separator) Unicode 7.0.0: General category Cf
     (Format).'''
    ],
    [[0x1E9C, 0x1E9D, 0x1E9F],
     [('lower', True), ('upper', False), ('tolower', False),
      ('toupper', False), ('totitle', False)],
     '''ẜ 0x1e9c LATIN SMALL LETTER LONG S WITH DIAGONAL STROKE,
     ẝ 0x1e9d LATIN SMALL LETTER LONG S WITH HIGH STROKE,
     ẟ 0x1e9f LATIN SMALL LETTER DELTA. These are “Lowercase”
     according to DerivedCoreProperties.txt but no upper case versions
     exist.'''
    ],
    [[0x1E9E],
     [('lower', False), ('upper', True), ('tolower', True),
      ('toupper', False), ('totitle', False)],
     '''0x1E9E ẞ LATIN CAPITAL LETTER SHARP S This is “Uppercase”
     according to DerivedCoreProperties.txt and the lower case
     version is 0x00DF ß LATIN SMALL LETTER SHARP S.'''
    ],
    [[0x2188],
     [('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''0x2188 ROMAN NUMERAL ONE HUNDRED THOUSAND.  This is “Alphabetic”
     according to DerivedCoreProperties.txt. In glibc’s old
     LC_CTYPE, it was in “lower”, which seems to be a
     mistake. It is not “Lowercase” in
     DerivedCoreProperties.txt and does not have case mappings
     in UnicodeData.txt either.'''
    ],
    [[0x2C71, 0x2C74, (0x2C77, 0x2C7A)],
            [('alpha', True), ('lower', True), ('upper', False),
             ('tolower', False), ('toupper', False), ('totitle', False)],
            '''These are Latin small letters which were not in Unicode 5.0.0
            but are in Unicode 7.0.0. According to
            DerivedCoreProperties.txt they are “Lowercase”. But no
            uppercase versions exist.  They have apparently been added
            manually to glibc’s old LC_CTYPE.'''
    ],
    [[0xA730, 0xA731],
            [('alpha', True), ('lower', True), ('upper', False),
             ('tolower', False), ('toupper', False), ('totitle', False)],
            '''These are Latin small “capital” letters which were not in
            Unicode 5.0.0 but are in Unicode 7.0.0. According to
            DerivedCoreProperties.txt they are “Lowercase”. But no
            uppercase versions exist.  They have apparently been added
            manually to glibc’s old LC_CTYPE.'''
    ],
    [[(0xA771, 0xA778)],
     [('alpha', True), ('lower', True), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''These are Latin small (or small “capital”) letters which
     were not in Unicodee 5.0.0 but are in Unicode 7.0.0. According to
     DerivedCoreProperties.txt they are “Lowercase”. But no
     uppercase versions exist.  They have apparently been added
     manually to glibc’s old LC_CTYPE.'''
    ],
    [[0x0375],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''“0375;GREEK LOWER NUMERAL SIGN;Sk;0;ON;;;;;N;;;;;”.  Has
     apparently been added manually to glibc’s old LC_CTYPE as
     “combining_level3”. That seems wrong, it is no combining
     character because it does not have one of the general
     categories Mn, Mc, or Me. According to
     DerivedCoreProperties.txt it is not “Alphabetic”.'''
    ],
    [[0x108D],
     [('combining', True), ('combining_level3', False),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''“108D;MYANMAR SIGN SHAN COUNCIL EMPHATIC
     TONE;Mn;220;NSM;;;;;N;;;;;”.  Has apparently been added
     manually to glibc’s old LC_CTYPE as
     “combining_level3”. That seems wrong, although it is a
     combining character because it has the general category
     Mn, it is not “combining_level3” because the canonical
     combining class value is 220 which is >= 200. According to
     gen-unicode-ctype.c, “combining_level3” needs a
     canonical combining class value < 200. According to
     DerivedCoreProperties.txt it was not “Alphabetic”
     until Unicode 11.0.0 but in 12.0.0 it became “Alphabetic”.'''
    ],
    [[0x06DE],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     ''' UnicodeData.txt 5.0.0: “06DE;ARABIC START OF RUB EL
     HIZB;Me;0;NSM;;;;;N;;;;;”; UnicodeData.txt 7.0.0:
     “06DE;ARABIC START OF RUB EL
     HIZB;So;0;ON;;;;;N;;;;;”. I.e. this used to be a
     combining character in Unicode 5.0.0 but not anymore in
     7.0.0. According to DerivedCoreProperties.txt it is not
     “Alphabetic”.'''
    ],
    [[0x0BD0],
     [('combining', False), ('combining_level3', False),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''Not in UnicodeData.txt 5.0.0.  UnicodeData.txt 7.0.0:
     “0BD0;TAMIL OM;Lo;0;L;;;;;N;;;;;”.  Apparently manually added to
     “combining” and “combining_level3” in glibc’s old
     LC_CTYPE. That seems wrong.  According to
     DerivedCoreProperties.txt it is “Alphabetic”.'''
    ],
    [[0x103F],
     [('combining', False), ('combining_level3', False),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''Not in UnicodeData.txt 5.0.0.  UnicodeData.txt 7.0.0:
     “103F;MYANMAR LETTER GREAT SA;Lo;0;L;;;;;N;;;;;”.
     Apparently manually added to “combining” and
     “combining_level3” in glibc’s old LC_CTYPE. That seems
     wrong.  According to DerivedCoreProperties.txt it is
     “Alphabetic”.'''
    ],
    [[(0x0901, 0x0903)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''These have general category “Mn” i.e. these are combining
     characters (both in UnicodeData.txt 5.0.0 and 7.0.0):
     “0901;DEVANAGARI SIGN CANDRABINDU;Mn;0;NSM;;;;;N;;;;;”,
     ”0902;DEVANAGARI SIGN ANUSVARA;Mn;0;NSM;;;;;N;;;;;”,
     “0903;DEVANAGARI SIGN VISARGA;Mc;0;L;;;;;N;;;;;”.
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x093C],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''UnicodeData.txt (5.0.0 and 7.0.0): “093C;DEVANAGARI SIGN
     NUKTA;Mn;7;NSM;;;;;N;;;;;” According to
     DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”. glibc’s old LC_TYPE has this in “alpha”.'''
    ],
    [[(0x093E, 0x093F)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''These have general category “Mc” i.e. these are combining
     characters (both in UnicodeData.txt 5.0.0 and 7.0.0):
     “093E;DEVANAGARI VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “093F;DEVANAGARI VOWEL SIGN I;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0x0940, 0x094C)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''These are all combining
     characters (“Mc” or “Mn” both in UnicodeData.txt 5.0.0 and 7.0.0).
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x094D],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''Combining character, both in UnicodeData.txt 5.0.0 and 7.0.0.
     “094D;DEVANAGARI SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) it is *not*
     “Alphabetic”.'''
    ],
    [[(0x0951, 0x0954)],
     [('combining', True), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''Combining characters, both in UnicodeData.txt 5.0.0 and 7.0.0.
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0962, 0x0963), (0x0981, 0x0983)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''Combining characters, both in UnicodeData.txt 5.0.0 and 7.0.0.
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x09BC],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “09BC;BENGALI SIGN NUKTA;Mn;7;NSM;;;;;N;;;;;”
     Combining character, both in UnicodeData.txt 5.0.0 and 7.0.0.
     According to DerivedCoreProperties.txt (7.0.0) it is *not*
     “Alphabetic”.'''
    ],
    [[(0x09BE, 0x09BF), (0x09C0, 0x09C4), (0x09C7, 0x09C8),
      (0x09CB, 0x09CC)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “09BE;BENGALI VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “09BF;BENGALI VOWEL SIGN I;Mc;0;L;;;;;N;;;;;”
     “09C0;BENGALI VOWEL SIGN II;Mc;0;L;;;;;N;;;;;”
     “09C1;BENGALI VOWEL SIGN U;Mn;0;NSM;;;;;N;;;;;”
     “09C2;BENGALI VOWEL SIGN UU;Mn;0;NSM;;;;;N;;;;;”
     “09C3;BENGALI VOWEL SIGN VOCALIC R;Mn;0;NSM;;;;;N;;;;;”
     “09C4;BENGALI VOWEL SIGN VOCALIC RR;Mn;0;NSM;;;;;N;;;;;”
     “09C7;BENGALI VOWEL SIGN E;Mc;0;L;;;;;N;;;;;”
     “09C8;BENGALI VOWEL SIGN AI;Mc;0;L;;;;;N;;;;;”
     “09CB;BENGALI VOWEL SIGN O;Mc;0;L;09C7 09BE;;;;N;;;;;”
     “09CC;BENGALI VOWEL SIGN AU;Mc;0;L;09C7 09D7;;;;N;;;;;”
     Combining characters, both in UnicodeData.txt 5.0.0 and 7.0.0.
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x09CD],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “09CD;BENGALI SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     Combining character, both in UnicodeData.txt 5.0.0 and 7.0.0.
     According to DerivedCoreProperties.txt (7.0.0) it is *not*
     “Alphabetic”.'''
    ],
    [[0x09D7, (0x09E2, 0x09E3)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''Combining characters, both in UnicodeData.txt 5.0.0 and 7.0.0.
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x09F2, 0x09F3],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “09F2;BENGALI RUPEE MARK;Sc;0;ET;;;;;N;;;;;”
     “09F3;BENGALI RUPEE SIGN;Sc;0;ET;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x09F4, 0x09FA)],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “09F4;BENGALI CURRENCY NUMERATOR ONE;No;0;L;;;;1/16;N;;;;;”
     “09F5;BENGALI CURRENCY NUMERATOR TWO;No;0;L;;;;1/8;N;;;;;”
     “09F6;BENGALI CURRENCY NUMERATOR THREE;No;0;L;;;;3/16;N;;;;;”
     “09F7;BENGALI CURRENCY NUMERATOR FOUR;No;0;L;;;;1/4;N;;;;;”
     “09F8;BENGALI CURRENCY NUMERATOR ONE LESS THAN THE DENOMINATOR;
     No;0;L;;;;3/4;N;;;;;”
     “09F9;BENGALI CURRENCY DENOMINATOR SIXTEEN;No;0;L;;;;16;N;;;;;”
     “09FA;BENGALI ISSHAR;So;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0A01, 0x0A03)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0A01;GURMUKHI SIGN ADAK BINDI;Mn;0;NSM;;;;;N;;;;;”
     “0A02;GURMUKHI SIGN BINDI;Mn;0;NSM;;;;;N;;;;;”
     “0A03;GURMUKHI SIGN VISARGA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0A3C],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0A3C;GURMUKHI SIGN NUKTA;Mn;7;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0A3E, 0x0A40), (0x0A41, 0x0A42), (0x0A47, 0x0A48),
      (0x0A4B, 0x0A4C)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0A3E;GURMUKHI VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “0A3F;GURMUKHI VOWEL SIGN I;Mc;0;L;;;;;N;;;;;”
     “0A40;GURMUKHI VOWEL SIGN II;Mc;0;L;;;;;N;;;;;”
     “0A41;GURMUKHI VOWEL SIGN U;Mn;0;NSM;;;;;N;;;;;”
     “0A42;GURMUKHI VOWEL SIGN UU;Mn;0;NSM;;;;;N;;;;;”
     “0A47;GURMUKHI VOWEL SIGN EE;Mn;0;NSM;;;;;N;;;;;”
     “0A48;GURMUKHI VOWEL SIGN AI;Mn;0;NSM;;;;;N;;;;;”
     “0A4B;GURMUKHI VOWEL SIGN OO;Mn;0;NSM;;;;;N;;;;;”
     “0A4C;GURMUKHI VOWEL SIGN AU;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0A4D],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0A4D;GURMUKHI SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[0x0A51, (0x0A70, 0x0A71), 0x0A75, (0x0A81, 0x0A83)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0A4D;GURMUKHI SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     “0A70;GURMUKHI TIPPI;Mn;0;NSM;;;;;N;;;;;”
     “0A71;GURMUKHI ADDAK;Mn;0;NSM;;;;;N;;;;;”
     “0A75;GURMUKHI SIGN YAKASH;Mn;0;NSM;;;;;N;;;;;”
     “0A81;GUJARATI SIGN CANDRABINDU;Mn;0;NSM;;;;;N;;;;;”
     “0A82;GUJARATI SIGN ANUSVARA;Mn;0;NSM;;;;;N;;;;;”
     “0A83;GUJARATI SIGN VISARGA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0ABC],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0ABC;GUJARATI SIGN NUKTA;Mn;7;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0ABE, 0x0AC5), (0x0AC7, 0x0AC9), (0x0ACB, 0x0ACC)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0ABE;GUJARATI VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “0ABF;GUJARATI VOWEL SIGN I;Mc;0;L;;;;;N;;;;;”
     “0AC0;GUJARATI VOWEL SIGN II;Mc;0;L;;;;;N;;;;;”
     “0AC1;GUJARATI VOWEL SIGN U;Mn;0;NSM;;;;;N;;;;;”
     “0AC2;GUJARATI VOWEL SIGN UU;Mn;0;NSM;;;;;N;;;;;”
     “0AC3;GUJARATI VOWEL SIGN VOCALIC R;Mn;0;NSM;;;;;N;;;;;”
     “0AC4;GUJARATI VOWEL SIGN VOCALIC RR;Mn;0;NSM;;;;;N;;;;;”
     “0AC5;GUJARATI VOWEL SIGN CANDRA E;Mn;0;NSM;;;;;N;;;;;”
     “0AC7;GUJARATI VOWEL SIGN E;Mn;0;NSM;;;;;N;;;;;”
     “0AC8;GUJARATI VOWEL SIGN AI;Mn;0;NSM;;;;;N;;;;;”
     “0AC9;GUJARATI VOWEL SIGN CANDRA O;Mc;0;L;;;;;N;;;;;”
     “0ACB;GUJARATI VOWEL SIGN O;Mc;0;L;;;;;N;;;;;”
     “0ACC;GUJARATI VOWEL SIGN AU;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0ACD],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0ACD;GUJARATI SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0AE2, 0x0AE3)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0AE2;GUJARATI VOWEL SIGN VOCALIC L;Mn;0;NSM;;;;;N;;;;;”
     “0AE3;GUJARATI VOWEL SIGN VOCALIC LL;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0AF1],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0AF1;GUJARATI RUPEE SIGN;Sc;0;ET;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0B01, 0x0B03)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B01;ORIYA SIGN CANDRABINDU;Mn;0;NSM;;;;;N;;;;;”
     “0B02;ORIYA SIGN ANUSVARA;Mc;0;L;;;;;N;;;;;”
     “0B03;ORIYA SIGN VISARGA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0B3C],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B3C;ORIYA SIGN NUKTA;Mn;7;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0B3E, 0x0B44), (0x0B47, 0x0B48), (0x0B4B, 0x0B4C)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B3E;ORIYA VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “0B3F;ORIYA VOWEL SIGN I;Mn;0;NSM;;;;;N;;;;;”
     “0B40;ORIYA VOWEL SIGN II;Mc;0;L;;;;;N;;;;;”
     “0B41;ORIYA VOWEL SIGN U;Mn;0;NSM;;;;;N;;;;;”
     “0B42;ORIYA VOWEL SIGN UU;Mn;0;NSM;;;;;N;;;;;”
     “0B43;ORIYA VOWEL SIGN VOCALIC R;Mn;0;NSM;;;;;N;;;;;”
     “0B44;ORIYA VOWEL SIGN VOCALIC RR;Mn;0;NSM;;;;;N;;;;;”
     “0B47;ORIYA VOWEL SIGN E;Mc;0;L;;;;;N;;;;;”
     “0B48;ORIYA VOWEL SIGN AI;Mc;0;L;0B47 0B56;;;;N;;;;;”
     “0B4B;ORIYA VOWEL SIGN O;Mc;0;L;0B47 0B3E;;;;N;;;;;”
     “0B4C;ORIYA VOWEL SIGN AU;Mc;0;L;0B47 0B57;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0B4D],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B4D;ORIYA SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0B56, 0x0B57), (0x0B62, 0x0B63)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B56;ORIYA AI LENGTH MARK;Mn;0;NSM;;;;;N;;;;;”
     “0B57;ORIYA AU LENGTH MARK;Mc;0;L;;;;;N;;;;;”
     “0B62;ORIYA VOWEL SIGN VOCALIC L;Mn;0;NSM;;;;;N;;;;;”
     “0B63;ORIYA VOWEL SIGN VOCALIC LL;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0B70],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B70;ORIYA ISSHAR;So;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[0x0B82],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0B82;TAMIL SIGN ANUSVARA;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0BBE, 0x0BC2), (0x0BC6, 0x0BC8), (0x0BCA, 0x0BCC)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0BBE;TAMIL VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “0BBF;TAMIL VOWEL SIGN I;Mc;0;L;;;;;N;;;;;”
     “0BC0;TAMIL VOWEL SIGN II;Mn;0;NSM;;;;;N;;;;;”
     “0BC1;TAMIL VOWEL SIGN U;Mc;0;L;;;;;N;;;;;”
     “0BC2;TAMIL VOWEL SIGN UU;Mc;0;L;;;;;N;;;;;”
     “0BC6;TAMIL VOWEL SIGN E;Mc;0;L;;;;;N;;;;;”
     “0BC7;TAMIL VOWEL SIGN EE;Mc;0;L;;;;;N;;;;;”
     “0BC8;TAMIL VOWEL SIGN AI;Mc;0;L;;;;;N;;;;;”
     “0BCA;TAMIL VOWEL SIGN O;Mc;0;L;0BC6 0BBE;;;;N;;;;;”
     “0BCB;TAMIL VOWEL SIGN OO;Mc;0;L;0BC7 0BBE;;;;N;;;;;”
     “0BCC;TAMIL VOWEL SIGN AU;Mc;0;L;0BC6 0BD7;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0BCD],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0BCD;TAMIL SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[0x0BD7],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0BD7;TAMIL AU LENGTH MARK;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0BF0, 0x0BFA)],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0BF0;TAMIL NUMBER TEN;No;0;L;;;;10;N;;;;;”
     “0BF1;TAMIL NUMBER ONE HUNDRED;No;0;L;;;;100;N;;;;;”
     “0BF2;TAMIL NUMBER ONE THOUSAND;No;0;L;;;;1000;N;;;;;”
     “0BF3;TAMIL DAY SIGN;So;0;ON;;;;;N;;;;;”
     “0BF4;TAMIL MONTH SIGN;So;0;ON;;;;;N;;;;;”
     “0BF5;TAMIL YEAR SIGN;So;0;ON;;;;;N;;;;;”
     “0BF6;TAMIL DEBIT SIGN;So;0;ON;;;;;N;;;;;”
     “0BF7;TAMIL CREDIT SIGN;So;0;ON;;;;;N;;;;;”
     “0BF8;TAMIL AS ABOVE SIGN;So;0;ON;;;;;N;;;;;”
     “0BF9;TAMIL RUPEE SIGN;Sc;0;ET;;;;;N;;;;;”
     “0BFA;TAMIL NUMBER SIGN;So;0;ON;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) this is *not*
     “Alphabetic”.'''
    ],
    [[(0x0C01, 0x0C03)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0C01;TELUGU SIGN CANDRABINDU;Mc;0;L;;;;;N;;;;;”
     “0C02;TELUGU SIGN ANUSVARA;Mc;0;L;;;;;N;;;;;”
     “0C03;TELUGU SIGN VISARGA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0x0C3E, 0x0C44), (0x0C46, 0x0C48), (0x0C4A, 0x0C4C)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0C3E;TELUGU VOWEL SIGN AA;Mn;0;NSM;;;;;N;;;;;”
     “0C3F;TELUGU VOWEL SIGN I;Mn;0;NSM;;;;;N;;;;;”
     “0C40;TELUGU VOWEL SIGN II;Mn;0;NSM;;;;;N;;;;;”
     “0C41;TELUGU VOWEL SIGN U;Mc;0;L;;;;;N;;;;;”
     “0C42;TELUGU VOWEL SIGN UU;Mc;0;L;;;;;N;;;;;”
     “0C43;TELUGU VOWEL SIGN VOCALIC R;Mc;0;L;;;;;N;;;;;”
     “0C44;TELUGU VOWEL SIGN VOCALIC RR;Mc;0;L;;;;;N;;;;;”
     “0C46;TELUGU VOWEL SIGN E;Mn;0;NSM;;;;;N;;;;;”
     “0C47;TELUGU VOWEL SIGN EE;Mn;0;NSM;;;;;N;;;;;”
     “0C48;TELUGU VOWEL SIGN AI;Mn;0;NSM;0C46 0C56;;;;N;;;;;”
     “0C4A;TELUGU VOWEL SIGN O;Mn;0;NSM;;;;;N;;;;;”
     “0C4B;TELUGU VOWEL SIGN OO;Mn;0;NSM;;;;;N;;;;;”
     “0C4C;TELUGU VOWEL SIGN AU;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0C4D],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0C4D;TELUGU SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0C55, 0x0C56), (0x0C62, 0x0C63)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0C55;TELUGU LENGTH MARK;Mn;84;NSM;;;;;N;;;;;”
     “0C56;TELUGU AI LENGTH MARK;Mn;91;NSM;;;;;N;;;;;”
     “0C62;TELUGU VOWEL SIGN VOCALIC L;Mn;0;NSM;;;;;N;;;;;”
     “0C63;TELUGU VOWEL SIGN VOCALIC LL;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0x0C78, 0x0C7F)],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0C78;TELUGU FRACTION DIGIT ZERO FOR ODD POWERS OF FOUR;
     No;0;ON;;;;0;N;;;;;”
     “0C79;TELUGU FRACTION DIGIT ONE FOR ODD POWERS OF FOUR;
     No;0;ON;;;;1;N;;;;;”
     “0C7A;TELUGU FRACTION DIGIT TWO FOR ODD POWERS OF FOUR;
     No;0;ON;;;;2;N;;;;;”
     “0C7B;TELUGU FRACTION DIGIT THREE FOR ODD POWERS OF FOUR;
     No;0;ON;;;;3;N;;;;;”
     “0C7C;TELUGU FRACTION DIGIT ONE FOR EVEN POWERS OF FOUR;
     No;0;ON;;;;1;N;;;;;”
     “0C7D;TELUGU FRACTION DIGIT TWO FOR EVEN POWERS OF FOUR;
     No;0;ON;;;;2;N;;;;;”
     “0C7E;TELUGU FRACTION DIGIT THREE FOR EVEN POWERS OF FOUR;
     No;0;ON;;;;3;N;;;;;”
     “0C7F;TELUGU SIGN TUUMU;So;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0C82, 0x0C83)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0C81;KANNADA SIGN CANDRABINDU;Mn;0;NSM;;;;;N;;;;;”
     “0C82;KANNADA SIGN ANUSVARA;Mc;0;L;;;;;N;;;;;”
     “0C83;KANNADA SIGN VISARGA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0CBC],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0CBC;KANNADA SIGN NUKTA;Mn;7;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0x0CBE, 0x0CC4), (0x0CC6, 0x0CC8), (0x0CCA, 0x0CCC)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0CBE;KANNADA VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “0CBF;KANNADA VOWEL SIGN I;Mn;0;L;;;;;N;;;;;”
     “0CC0;KANNADA VOWEL SIGN II;Mc;0;L;0CBF 0CD5;;;;N;;;;;”
     “0CC1;KANNADA VOWEL SIGN U;Mc;0;L;;;;;N;;;;;”
     “0CC2;KANNADA VOWEL SIGN UU;Mc;0;L;;;;;N;;;;;”
     “0CC3;KANNADA VOWEL SIGN VOCALIC R;Mc;0;L;;;;;N;;;;;”
     “0CC4;KANNADA VOWEL SIGN VOCALIC RR;Mc;0;L;;;;;N;;;;;”
     “0CC6;KANNADA VOWEL SIGN E;Mn;0;L;;;;;N;;;;;”
     “0CC7;KANNADA VOWEL SIGN EE;Mc;0;L;0CC6 0CD5;;;;N;;;;;”
     “0CC8;KANNADA VOWEL SIGN AI;Mc;0;L;0CC6 0CD6;;;;N;;;;;”
     “0CCA;KANNADA VOWEL SIGN O;Mc;0;L;0CC6 0CC2;;;;N;;;;;”
     “0CCB;KANNADA VOWEL SIGN OO;Mc;0;L;0CCA 0CD5;;;;N;;;;;”
     “0CCC;KANNADA VOWEL SIGN AU;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0CCD],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0CCD;KANNADA SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0CD5, 0x0CD6), (0x0CE2, 0x0CE3)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     0CD5;KANNADA LENGTH MARK;Mc;0;L;;;;;N;;;;;
     0CD6;KANNADA AI LENGTH MARK;Mc;0;L;;;;;N;;;;;
     0CE2;KANNADA VOWEL SIGN VOCALIC L;Mn;0;NSM;;;;;N;;;;;
     0CE3;KANNADA VOWEL SIGN VOCALIC LL;Mn;0;NSM;;;;;N;;;;;
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0x0D02, 0x0D03), (0x0D3E, 0x0D44), (0x0D46, 0x0D48),
      (0x0D4A, 0x0D4C)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0D02;MALAYALAM SIGN ANUSVARA;Mc;0;L;;;;;N;;;;;”
     “0D03;MALAYALAM SIGN VISARGA;Mc;0;L;;;;;N;;;;;”
     “0D3E;MALAYALAM VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;”
     “0D3F;MALAYALAM VOWEL SIGN I;Mc;0;L;;;;;N;;;;;”
     “0D40;MALAYALAM VOWEL SIGN II;Mc;0;L;;;;;N;;;;;”
     “0D41;MALAYALAM VOWEL SIGN U;Mn;0;NSM;;;;;N;;;;;”
     “0D42;MALAYALAM VOWEL SIGN UU;Mn;0;NSM;;;;;N;;;;;”
     “0D43;MALAYALAM VOWEL SIGN VOCALIC R;Mn;0;NSM;;;;;N;;;;;”
     “0D44;MALAYALAM VOWEL SIGN VOCALIC RR;Mn;0;NSM;;;;;N;;;;;”
     “0D46;MALAYALAM VOWEL SIGN E;Mc;0;L;;;;;N;;;;;”
     “0D47;MALAYALAM VOWEL SIGN EE;Mc;0;L;;;;;N;;;;;”
     “0D48;MALAYALAM VOWEL SIGN AI;Mc;0;L;;;;;N;;;;;”
     “0D4A;MALAYALAM VOWEL SIGN O;Mc;0;L;0D46 0D3E;;;;N;;;;;”
     “0D4B;MALAYALAM VOWEL SIGN OO;Mc;0;L;0D47 0D3E;;;;N;;;;;”
     “0D4C;MALAYALAM VOWEL SIGN AU;Mc;0;L;0D46 0D57;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0D4D],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0D4D;MALAYALAM SIGN VIRAMA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0D57, (0x0D62, 0x0D63)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0D57;MALAYALAM AU LENGTH MARK;Mc;0;L;;;;;N;;;;;”
     “0D62;MALAYALAM VOWEL SIGN VOCALIC L;Mn;0;NSM;;;;;N;;;;;”
     “0D63;MALAYALAM VOWEL SIGN VOCALIC LL;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0x0D70, 0x0D79)],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0D70;MALAYALAM NUMBER TEN;No;0;L;;;;10;N;;;;;”
     “0D71;MALAYALAM NUMBER ONE HUNDRED;No;0;L;;;;100;N;;;;;”
     “0D72;MALAYALAM NUMBER ONE THOUSAND;No;0;L;;;;1000;N;;;;;”
     “0D73;MALAYALAM FRACTION ONE QUARTER;No;0;L;;;;1/4;N;;;;;”
     “0D74;MALAYALAM FRACTION ONE HALF;No;0;L;;;;1/2;N;;;;;”
     “0D75;MALAYALAM FRACTION THREE QUARTERS;No;0;L;;;;3/4;N;;;;;”
     “0D79;MALAYALAM DATE MARK;So;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0D82, 0x0D83)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0D82;SINHALA SIGN ANUSVARAYA;Mc;0;L;;;;;N;;;;;”
     “0D83;SINHALA SIGN VISARGAYA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0DCA],
     [('combining', True), ('combining_level3', True),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0DCA;SINHALA SIGN AL-LAKUNA;Mn;9;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0x0DCF, 0x0DD4), 0x0DD6, (0x0DD8, 0x0DDF), (0x0DF2, 0x0DF3)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0DCF;SINHALA VOWEL SIGN AELA-PILLA;Mc;0;L;;;;;N;;;;;”
     “0DD0;SINHALA VOWEL SIGN KETTI AEDA-PILLA;Mc;0;L;;;;;N;;;;;”
     “0DD1;SINHALA VOWEL SIGN DIGA AEDA-PILLA;Mc;0;L;;;;;N;;;;;”
     “0DD2;SINHALA VOWEL SIGN KETTI IS-PILLA;Mn;0;NSM;;;;;N;;;;;”
     “0DD3;SINHALA VOWEL SIGN DIGA IS-PILLA;Mn;0;NSM;;;;;N;;;;;”
     “0DD4;SINHALA VOWEL SIGN KETTI PAA-PILLA;Mn;0;NSM;;;;;N;;;;;”
     “0DD6;SINHALA VOWEL SIGN DIGA PAA-PILLA;Mn;0;NSM;;;;;N;;;;;”
     “0DD8;SINHALA VOWEL SIGN GAETTA-PILLA;Mc;0;L;;;;;N;;;;;”
     “0DD9;SINHALA VOWEL SIGN KOMBUVA;Mc;0;L;;;;;N;;;;;”
     “0DDA;SINHALA VOWEL SIGN DIGA KOMBUVA;Mc;0;L;0DD9 0DCA;;;;N;;;;;”
     “0DDB;SINHALA VOWEL SIGN KOMBU DEKA;Mc;0;L;;;;;N;;;;;”
     “0DDC;SINHALA VOWEL SIGN KOMBUVA HAA AELA-PILLA;
     Mc;0;L;0DD9 0DCF;;;;N;;;;;”
     “0DDD;SINHALA VOWEL SIGN KOMBUVA HAA DIGA AELA-PILLA;
     Mc;0;L;0DDC 0DCA;;;;N;;;;;”
     “0DDE;SINHALA VOWEL SIGN KOMBUVA HAA GAYANUKITTA;
     Mc;0;L;0DD9 0DDF;;;;N;;;;;”
     “0DDF;SINHALA VOWEL SIGN GAYANUKITTA;Mc;0;L;;;;;N;;;;;”
     “0DF2;SINHALA VOWEL SIGN DIGA GAETTA-PILLA;Mc;0;L;;;;;N;;;;;”
     “0DF3;SINHALA VOWEL SIGN DIGA GAYANUKITTA;Mc;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[0x0DF4],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “0DF4;SINHALA PUNCTUATION KUNDDALIYA;Po;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0xA789, 0xA78A)],
     [('combining', False), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “A789;MODIFIER LETTER COLON;Sk;0;L;;;;;N;;;;;”
     “A78A;MODIFIER LETTER SHORT EQUALS SIGN;Sk;0;L;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ],
    [[(0xA926, 0xA92A)],
     [('combining', True), ('combining_level3', True),
      ('alpha', True), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “A926;KAYAH LI VOWEL UE;Mn;0;NSM;;;;;N;;;;;”
     “A927;KAYAH LI VOWEL E;Mn;0;NSM;;;;;N;;;;;”
     “A928;KAYAH LI VOWEL U;Mn;0;NSM;;;;;N;;;;;”
     “A929;KAYAH LI VOWEL EE;Mn;0;NSM;;;;;N;;;;;”
     “A92A;KAYAH LI VOWEL O;Mn;0;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are
     “Alphabetic”.'''
    ],
    [[(0xA92B, 0xA92D)],
     [('combining', True), ('combining_level3', False),
      ('alpha', False), ('lower', False), ('upper', False),
      ('tolower', False), ('toupper', False), ('totitle', False)],
     '''
     “A92B;KAYAH LI TONE PLOPHU;Mn;220;NSM;;;;;N;;;;;”
     “A92C;KAYAH LI TONE CALYA;Mn;220;NSM;;;;;N;;;;;”
     “A92D;KAYAH LI TONE CALYA PLOPHU;Mn;220;NSM;;;;;N;;;;;”
     According to DerivedCoreProperties.txt (7.0.0) these are *not*
     “Alphabetic”.'''
    ]
]
