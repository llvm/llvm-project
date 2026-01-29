//===-- Unittests for wctype classification utils -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/wctype/wctype_classification_utils.h"
#include "test/UnitTest/Test.h"

namespace {

// Some platform (like Windows) have a 16 bit wchar_t. We guard the cases that
// do not fit within 16 bits to prevent narrowing conversion and incorrect test
// results.
struct TestCase {
  uint32_t wc;
  const char *name;
  bool expected;
};

TEST(LlvmLibcWctypeClassificationUtilsTest, Lower) {
  TestCase cases[] = {// ASCII lowercase
                      {0x0061, "LATIN SMALL LETTER A", true},
                      {0x007A, "LATIN SMALL LETTER Z", true},

                      // ASCII uppercase
                      {0x0041, "LATIN CAPITAL LETTER A", false},
                      {0x005A, "LATIN CAPITAL LETTER Z", false},

                      // ASCII non-letters
                      {0x0030, "DIGIT ZERO", false},
                      {0x0020, "SPACE", false},
                      {0x0021, "EXCLAMATION MARK", false},

                      // Latin Extended lowercase
                      {0x00E0, "LATIN SMALL LETTER A WITH GRAVE", true},
                      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", true},
                      {0x00FF, "LATIN SMALL LETTER Y WITH DIAERESIS", true},

                      // Latin Extended uppercase
                      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", false},
                      {0x00C9, "LATIN CAPITAL LETTER E WITH ACUTE", false},

                      // Greek lowercase
                      {0x03B1, "GREEK SMALL LETTER ALPHA", true},
                      {0x03C9, "GREEK SMALL LETTER OMEGA", true},

                      // Greek uppercase
                      {0x0391, "GREEK CAPITAL LETTER ALPHA", false},
                      {0x03A9, "GREEK CAPITAL LETTER OMEGA", false},

                      // Cyrillic lowercase
                      {0x0430, "CYRILLIC SMALL LETTER A", true},
                      {0x044F, "CYRILLIC SMALL LETTER YA", true},

                      // Cyrillic uppercase
                      {0x0410, "CYRILLIC CAPITAL LETTER A", false},
                      {0x042F, "CYRILLIC CAPITAL LETTER YA", false},

                      // Caseless scripts
                      {0x05D0, "HEBREW LETTER ALEF", false},
                      {0x0627, "ARABIC LETTER ALEF", false},
                      {0x4E00, "CJK UNIFIED IDEOGRAPH-4E00", false}};

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::LOWER;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Upper) {
  TestCase cases[] = {
      // ASCII lowercase
      {0x0061, "LATIN SMALL LETTER A", false},
      {0x007A, "LATIN SMALL LETTER Z", false},

      // ASCII uppercase
      {0x0041, "LATIN CAPITAL LETTER A", true},
      {0x005A, "LATIN CAPITAL LETTER Z", true},

      // ASCII non-letters
      {0x0030, "DIGIT ZERO", false},
      {0x0020, "SPACE", false},
      {0x0021, "EXCLAMATION MARK", false},

      // Titlecase
      {0x01C5, "LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON", true},

      // Latin Extended lowercase
      {0x00E0, "LATIN SMALL LETTER A WITH GRAVE", false},
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x00FF, "LATIN SMALL LETTER Y WITH DIAERESIS", false},

      // Latin Extended uppercase
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", true},
      {0x00C9, "LATIN CAPITAL LETTER E WITH ACUTE", true},

      // Greek lowercase
      {0x03B1, "GREEK SMALL LETTER ALPHA", false},
      {0x03C9, "GREEK SMALL LETTER OMEGA", false},

      // Greek uppercase
      {0x0391, "GREEK CAPITAL LETTER ALPHA", true},
      {0x03A9, "GREEK CAPITAL LETTER OMEGA", true},

      // Cyrillic lowercase
      {0x0430, "CYRILLIC SMALL LETTER A", false},
      {0x044F, "CYRILLIC SMALL LETTER YA", false},

      // Cyrillic uppercase
      {0x0410, "CYRILLIC CAPITAL LETTER A", true},
      {0x042F, "CYRILLIC CAPITAL LETTER YA", true},

      // Caseless scripts
      {0x05D0, "HEBREW LETTER ALEF", false},
      {0x0627, "ARABIC LETTER ALEF", false},
      {0x4E00, "CJK UNIFIED IDEOGRAPH-4E00", false}};

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::UPPER;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Alpha) {
  TestCase cases[] = {
      // ASCII letters
      {0x0041, "LATIN CAPITAL LETTER A", true},
      {0x0061, "LATIN SMALL LETTER A", true},
      {0x005A, "LATIN CAPITAL LETTER Z", true},
      {0x007A, "LATIN SMALL LETTER Z", true},

      // ASCII non-letters
      {0x0030, "DIGIT ZERO", false},
      {0x0039, "DIGIT NINE", false},
      {0x0020, "SPACE", false},
      {0x0021, "EXCLAMATION MARK", false},
      {0x007E, "TILDE", false},

      // Modified letters
      {0x02B0, "MODIFIED LETTER SMALL H", true},

      // Latin Extended
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", true},
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", true},
      {0x00FF, "LATIN SMALL LETTER Y WITH DIAERESIS", true},

      // Greek
      {0x0391, "GREEK CAPITAL LETTER ALPHA", true},
      {0x03B1, "GREEK SMALL LETTER ALPHA", true},
      {0x03C9, "GREEK SMALL LETTER OMEGA", true},

      // Cyrillic
      {0x0410, "CYRILLIC CAPITAL LETTER A", true},
      {0x0430, "CYRILLIC SMALL LETTER A", true},
      {0x044F, "CYRILLIC SMALL LETTER YA", true},

      // Arabic
      {0x0627, "ARABIC LETTER ALEF", true},
      {0x0628, "ARABIC LETTER BEH", true},

      // CJK
      {0x4E00, "CJK UNIFIED IDEOGRAPH-4E00 (first)", true},
      {0x4E01, "CJK UNIFIED IDEOGRAPH-4E01", true},
      {0x9FFF, "CJK UNIFIED IDEOGRAPH-9FFF (last in BMP)", true},

      // Emoji and symbols
      {0x2764, "HEAVY BLACK HEART", false},

      // Special cases
      {0x0000, "NULL", false},
      {0xFFFD, "REPLACEMENT CHARACTER", false},

      // Roman numerals
      {0x2160, "ROMAN NUMERAL ONE", true},
      {0x2161, "ROMAN NUMERAL TWO", true},
      {0x2162, "ROMAN NUMERAL THREE", true},
      {0x2169, "ROMAN NUMERAL TEN", true},
      {0x216C, "ROMAN NUMERAL FIFTY", true},
      {0x216D, "ROMAN NUMERAL ONE HUNDRED", true},
      {0x216E, "ROMAN NUMERAL FIVE HUNDRED", true},
      {0x216F, "ROMAN NUMERAL ONE THOUSAND", true},

      // ASCII digits
      {0x0030, "DIGIT ZERO", false},
      {0x0031, "DIGIT ONE", false},

      // Non ASCII digits
      {0x0660, "ARABIC-INDIC DIGIT ZERO", true},
      {0x09e6, "BENGALI DIGIT ZERO", true},

      // Combining marks
      {0x0300, "COMBINING GRAVE ACCENT", false},

#if WCHAR_MAX > 0xFFFF
      {0x1F600, "GRINNING FACE", false},
      {0x20000, "CJK UNIFIED IDEOGRAPH-20000", true},
#endif

  };

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::ALPHA;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Punct) {
  TestCase cases[] = {// ASCII punctuation
                      {0x0021, "EXCLAMATION MARK", true},
                      {0x0022, "QUOTATION MARK", true},
                      {0x0023, "NUMBER SIGN", true},
                      {0x002C, "COMMA", true},
                      {0x002E, "FULL STOP", true},
                      {0x002F, "SOLIDUS", true},
                      {0x003A, "COLON", true},
                      {0x003B, "SEMICOLON", true},
                      {0x003F, "QUESTION MARK", true},
                      {0x0040, "COMMERCIAL AT", true},
                      {0x005B, "LEFT SQUARE BRACKET", true},
                      {0x005D, "RIGHT SQUARE BRACKET", true},
                      {0x007B, "LEFT CURLY BRACKET", true},
                      {0x007D, "RIGHT CURLY BRACKET", true},

                      // ASCII non-punctuation
                      {0x0041, "LATIN CAPITAL LETTER A", false},
                      {0x0061, "LATIN SMALL LETTER A", false},
                      {0x0030, "DIGIT ZERO", false},
                      {0x0020, "SPACE", false},

                      // Unicode punctuation
                      {0x00A1, "INVERTED EXCLAMATION MARK", true},
                      {0x00BF, "INVERTED QUESTION MARK", true},
                      {0x2013, "EN DASH", true},
                      {0x2014, "EM DASH", true},
                      {0x2018, "LEFT SINGLE QUOTATION MARK", true},
                      {0x2019, "RIGHT SINGLE QUOTATION MARK", true},
                      {0x201C, "LEFT DOUBLE QUOTATION MARK", true},
                      {0x201D, "RIGHT DOUBLE QUOTATION MARK", true},
                      {0x2026, "HORIZONTAL ELLIPSIS", true},
                      {0x2030, "PER MILLE SIGN", true},
                      {0x3001, "IDEOGRAPHIC COMMA", true},
                      {0x3002, "IDEOGRAPHIC FULL STOP", true},
                      {0xFF01, "FULLWIDTH EXCLAMATION MARK", true},
                      {0xFF1F, "FULLWIDTH QUESTION MARK", true},

                      // Symbols (treated as punct in C.UTF-8)
                      {0x00A9, "COPYRIGHT SIGN", true},
                      {0x20AC, "EURO SIGN", true},
                      {0x2764, "HEAVY BLACK HEART", true},
                      {0x002B, "PLUS SIGN", true},
                      {0x00B6, "PILCROW SIGN", true},
                      {0x00A7, "SECTION SIGN", true},
                      {0x2022, "BULLET", true},
                      {0x2023, "TRIANGULAR BULLET", true},
                      {0x2020, "DAGGER", true},
                      {0x2021, "DOUBLE DAGGER", true},

                      // Math symbols (treated as punct in C.UTF-8)
                      {0x00D7, "MULTIPLICATION SIGN", true},
                      {0x00F7, "DIVISION SIGN", true},
                      {0x2212, "MINUS SIGN", true},
                      {0x221E, "INFINITY", true}};

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::PUNCT;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Print) {
  TestCase cases[] = {
      // ASCII printable characters
      {0x0020, "SPACE", true},
      {0x0021, "EXCLAMATION MARK", true},
      {0x0030, "DIGIT ZERO", true},
      {0x0041, "LATIN CAPITAL LETTER A", true},
      {0x0061, "LATIN SMALL LETTER A", true},
      {0x007E, "TILDE", true},

      // ASCII control characters
      {0x0000, "NULL", false},
      {0x0009, "TAB", false},
      {0x000A, "LINE FEED", false},
      {0x000D, "CARRIAGE RETURN", false},
      {0x001F, "UNIT SEPARATOR", false},
      {0x007F, "DELETE", false},

      // Non ASCII printable
      {0x00A0, "NO-BREAK SPACE", true},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", true},
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", true},
      {0x00FF, "LATIN SMALL LETTER Y WITH DIAERESIS", true},
      {0x0391, "GREEK CAPITAL LETTER ALPHA", true},
      {0x03B1, "GREEK SMALL LETTER ALPHA", true},
      {0x0410, "CYRILLIC CAPITAL LETTER A", true},
      {0x0430, "CYRILLIC SMALL LETTER A", true},
      {0x0627, "ARABIC LETTER ALEF", true},
      {0x05D0, "HEBREW LETTER ALEF", true},
      {0x4E00, "CJK UNIFIED IDEOGRAPH-4E00", true},
      {0x9FFF, "CJK UNIFIED IDEOGRAPH-9FFF", true},
      {0x3042, "HIRAGANA LETTER A", true},
      {0x30A2, "KATAKANA LETTER A", true},
      {0xAC00, "HANGUL SYLLABLE GA", true},

      // Emoji and symbols
      {0x2764, "HEAVY BLACK HEART", true},

      // Punctuation
      {0x002E, "FULL STOP", true},
      {0x002C, "COMMA", true},
      {0x003A, "COLON", true},

      // C1 control characters
      {0x0080, "PADDING CHARACTER", false},
      {0x009F, "APPLICATION PROGRAM COMMAND", false},

      {0xFFFD, "REPLACEMENT CHARACTER", true},

      // Format characters
      {0x00AD, "SOFT HYPHEN", false},
      {0x200C, "ZERO WIDTH NON-JOINER", false},

      // Combining marks
      {0x0300, "COMBINING GRAVE ACCENT", true},

      // Private use area
      {0xE000, "PRIVATE USE AREA (first)", true},
      {0xF000, "PRIVATE USE AREA (last)", true},

#if WCHAR_MAX > 0xFFFF
      {0x10FFFD, "SUPPLEMENTARY PRIVATE USE AREA B", true},
      {0x1F600, "GRINNING FACE", true},
#endif
  };

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::PRINT;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Control) {
  TestCase cases[] = {// ASCII control characters
                      {0x0000, "NULL", true},
                      {0x0001, "START OF HEADING", true},
                      {0x0009, "TAB", true},
                      {0x000A, "LINE FEED", true},
                      {0x000D, "CARRIAGE RETURN", true},
                      {0x001B, "ESCAPE", true},
                      {0x001F, "UNIT SEPARATOR", true},

                      // ASCII printable characters
                      {0x0020, "SPACE", false},
                      {0x0021, "EXCLAMATION MARK", false},
                      {0x0030, "DIGIT ZERO", false},
                      {0x0041, "LATIN CAPITAL LETTER A", false},
                      {0x0061, "LATIN SMALL LETTER A", false},
                      {0x007E, "TILDE", false},

                      // DELETE character
                      {0x007F, "DELETE", true},

                      // C1 control characters
                      {0x0080, "PADDING CHARACTER", true},
                      {0x0081, "HIGH OCTET PRESET", true},
                      {0x0090, "DEVICE CONTROL STRING", true},
                      {0x009F, "APPLICATION PROGRAM COMMAND", true},

                      // Non-control characters after C1 range
                      {0x00A0, "NO-BREAK SPACE", false},
                      {0x00A1, "INVERTED EXCLAMATION MARK", false},
                      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", false},
                      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},

                      // Letters
                      {0x0391, "GREEK CAPITAL LETTER ALPHA", false},
                      {0x0410, "CYRILLIC CAPITAL LETTER A", false},
                      {0x4E00, "CJK UNIFIED IDEOGRAPH-4E00", false}};

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::CNTRL;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Space) {
  TestCase cases[] = {// ASCII whitespace
                      {0x0020, "SPACE", true},
                      {0x0009, "CHARACTER TABULATION (TAB)", true},
                      {0x000A, "LINE FEED", true},
                      {0x000B, "LINE TABULATION", true},
                      {0x000C, "FORM FEED", true},
                      {0x000D, "CARRIAGE RETURN", true},

                      // ASCII non-whitespace
                      {0x0041, "LATIN CAPITAL LETTER A", false},
                      {0x0030, "DIGIT ZERO", false},
                      {0x0021, "EXCLAMATION MARK", false},

                      // Unicode whitespace
                      {0x1680, "OGHAM SPACE MARK", true},
                      {0x2000, "EN QUAD", true},
                      {0x2001, "EM QUAD", true},
                      {0x2002, "EN SPACE", true},
                      {0x2003, "EM SPACE", true},
                      {0x2004, "THREE-PER-EM SPACE", true},
                      {0x2005, "FOUR-PER-EM SPACE", true},
                      {0x2006, "SIX-PER-EM SPACE", true},
                      {0x2008, "PUNCTUATION SPACE", true},
                      {0x2009, "THIN SPACE", true},
                      {0x200A, "HAIR SPACE", true},
                      {0x2028, "LINE SEPARATOR", true},
                      {0x2029, "PARAGRAPH SEPARATOR", true},
                      {0x205F, "MEDIUM MATHEMATICAL SPACE", true},
                      {0x3000, "IDEOGRAPHIC SPACE", true},

                      // Unicode non-whitespace
                      {0x202F, "NARROW NO-BREAK SPACE", false},
                      {0x0085, "NEXT LINE", false},
                      {0x00A0, "NO-BREAK SPACE", false},
                      {0x2007, "FIGURE SPACE", false},
                      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
                      {0x2764, "HEAVY BLACK HEART", false}};

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::SPACE;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Blank) {
  TestCase cases[] = {// Blank characters
                      {0x0020, "SPACE", true},
                      {0x0009, "CHARACTER TABULATION (TAB)", true},

                      // Non-blank whitespace
                      {0x000A, "LINE FEED", false},
                      {0x000D, "CARRIAGE RETURN", false},
                      {0x000B, "LINE TABULATION", false},
                      {0x000C, "FORM FEED", false},

                      // Unicode blank characters
                      {0x1680, "OGHAM SPACE MARK", true},
                      {0x2000, "EN QUAD", true},
                      {0x2001, "EM QUAD", true},
                      {0x2002, "EN SPACE", true},
                      {0x2003, "EM SPACE", true},
                      {0x2004, "THREE-PER-EM SPACE", true},
                      {0x2005, "FOUR-PER-EM SPACE", true},
                      {0x2006, "SIX-PER-EM SPACE", true},
                      {0x2008, "PUNCTUATION SPACE", true},
                      {0x2009, "THIN SPACE", true},
                      {0x200A, "HAIR SPACE", true},
                      {0x3000, "IDEOGRAPHIC SPACE", true},

                      // Non-blank characters
                      {0x0041, "LATIN CAPITAL LETTER A", false},
                      {0x0030, "DIGIT ZERO", false},
                      {0x0021, "EXCLAMATION MARK", false},
                      {0x00A0, "NO-BREAK SPACE", false},
                      {0x2007, "FIGURE SPACE", false},
                      {0x202F, "NARROW NO-BREAK SPACE", false},
                      {0x205F, "MEDIUM MATHEMATICAL SPACE", true},
                      {0x2028, "LINE SEPARATOR", false}};

  for (const auto &tc : cases) {
    bool res = LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc)) &
               LIBC_NAMESPACE::PropertyFlag::BLANK;
    EXPECT_EQ(res, tc.expected) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, InvalidCodepoints) {
  struct InvalidTestCase {
    uint32_t wc;
    const char *name;
  };

  InvalidTestCase cases[] = {
      // Surrogate pair range
      {0xD800, "HIGH SURROGATE START"}, {0xD900, "HIGH SURROGATE MIDDLE"},
      {0xDBFF, "HIGH SURROGATE END"},   {0xDC00, "LOW SURROGATE START"},
      {0xDD00, "LOW SURROGATE MIDDLE"}, {0xDFFF, "LOW SURROGATE END"},

#if WCHAR_MAX > 0xFFFF
      {0x110000, "Beyond max Unicode"},
#endif
  };

  for (const auto &tc : cases) {
    uint8_t props =
        LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc));
    EXPECT_EQ(props, uint8_t{0}) << tc.name << "\n";
  }
}

TEST(LlvmLibcWctypeClassificationUtilsTest, Noncharacters) {
  struct NoncharacterTestCase {
    uint32_t wc;
    const char *name;
  };

  NoncharacterTestCase cases[] = {
      // BMP noncharacters
      {0xFFFE, "BMP NONCHARACTER U+FFFE"},
      {0xFFFF, "BMP NONCHARACTER U+FFFF"},

      // Arabic Presentation Forms noncharacters
      {0xFDD0, "NONCHARACTER U+FDD0"},
      {0xFDD5, "NONCHARACTER U+FDD5"},

#if WCHAR_MAX > 0xFFFF
      // Supplementary plane noncharacters
      {0x1FFFE, "PLANE 1 NONCHARACTER"},
      {0x2FFFE, "PLANE 2 NONCHARACTER"},
      {0x3FFFE, "PLANE 3 NONCHARACTER"},
      {0x10FFFE, "PLANE 16 NONCHARACTER"},
      {0x10FFFF, "PLANE 16 NONCHARACTER"},
#endif
  };

  for (const auto &tc : cases) {
    uint8_t props =
        LIBC_NAMESPACE::lookup_properties(static_cast<wchar_t>(tc.wc));
    EXPECT_EQ(props, uint8_t{0}) << tc.name << "\n";
  }
}

} // namespace
