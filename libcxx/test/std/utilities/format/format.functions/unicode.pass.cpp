//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// This version runs the test when the platform has Unicode support.
// UNSUPPORTED: libcpp-has-no-unicode

// TODO FMT Investigate Windows and AIX issues.
// UNSUPPORTED msvc, target={{.+}}-windows-gnu
// UNSUPPORTED: LIBCXX-AIX-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <format>

// Tests the Unicode width support of the standard format specifiers.
// It tests [format.string.std]/8 - 11:
// - Properly determining the estimated with of a unicode string.
// - Properly truncating to the wanted maximum width.

// More specific extended grapheme cluster boundary rules are tested in
// test/libcxx/utilities/format/format.string/format.string.std/extended_grapheme_cluster.pass.cpp
// this test is based on test data provided by the Unicode Consortium.

#include <format>
#include <cassert>
#include <vector>

#include "make_string.h"
#include "test_macros.h"
#include "string_literal.h"
#include "test_format_string.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <iostream>
#  include <type_traits>
#endif

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template < class CharT, class... Args>
void check(std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
#ifndef TEST_HAS_NO_LOCALIZATION
  if constexpr (std::same_as<CharT, char>)
    if (out != expected)
      std::cerr << "\nFormat string   " << fmt.get() << "\nExpected output " << expected << "\nActual output   " << out
                << '\n';
#endif
  assert(out == expected);
};

template <class CharT>
static void test_single_code_point_fill() {
  //*** 1-byte code points ***
  check(SV("* *"), SV("{:*^3}"), SV(" "));
  check(SV("*~*"), SV("{:*^3}"), SV("~"));

  //*** 2-byte code points ***
  check(SV("*\u00a1*"), SV("{:*^3}"), SV("\u00a1")); // INVERTED EXCLAMATION MARK
  check(SV("*\u07ff*"), SV("{:*^3}"), SV("\u07ff")); // NKO TAMAN SIGN

  //*** 3-byte code points ***
  check(SV("*\u0800*"), SV("{:*^3}"), SV("\u0800")); // SAMARITAN LETTER ALAF
  check(SV("*\ufffd*"), SV("{:*^3}"), SV("\ufffd")); // REPLACEMENT CHARACTER

  // 2 column ranges
  check(SV("*\u1100*"), SV("{:*^4}"), SV("\u1100")); // HANGUL CHOSEONG KIYEOK
  check(SV("*\u115f*"), SV("{:*^4}"), SV("\u115f")); // HANGUL CHOSEONG FILLER

  check(SV("*\u2329*"), SV("{:*^4}"), SV("\u2329")); // LEFT-POINTING ANGLE BRACKET
  check(SV("*\u232a*"), SV("{:*^4}"), SV("\u232a")); // RIGHT-POINTING ANGLE BRACKET

  check(SV("*\u2e80*"), SV("{:*^4}"), SV("\u2e80")); // CJK RADICAL REPEAT
  check(SV("*\u303e*"), SV("{:*^4}"), SV("\u303e")); // IDEOGRAPHIC VARIATION INDICATOR

  check(SV("*\u3040*"), SV("{:*^4}"), SV("\u3040")); // U+3041 HIRAGANA LETTER SMALL A
  check(SV("*\ua4cf*"), SV("{:*^4}"), SV("\ua4cf")); // U+A4D0 LISU LETTER BA

  check(SV("*\uac00*"), SV("{:*^4}"), SV("\uac00")); // <Hangul Syllable, First>
  check(SV("*\ud7a3*"), SV("{:*^4}"), SV("\ud7a3")); // Hangul Syllable Hih

  check(SV("*\uf900*"), SV("{:*^4}"), SV("\uf900")); // CJK COMPATIBILITY IDEOGRAPH-F900
  check(SV("*\ufaff*"), SV("{:*^4}"), SV("\ufaff")); // U+FB00 LATIN SMALL LIGATURE FF

  check(SV("*\ufe10*"), SV("{:*^4}"), SV("\ufe10")); // PRESENTATION FORM FOR VERTICAL COMMA
  check(SV("*\ufe19*"), SV("{:*^4}"), SV("\ufe19")); // PRESENTATION FORM FOR VERTICAL HORIZONTAL ELLIPSIS

  check(SV("*\ufe30*"), SV("{:*^4}"), SV("\ufe30")); // PRESENTATION FORM FOR VERTICAL TWO DOT LEADER
  check(SV("*\ufe6f*"), SV("{:*^4}"), SV("\ufe6f")); // U+FE70 ARABIC FATHATAN ISOLATED FORM

  check(SV("*\uff00*"), SV("{:*^4}"), SV("\uff00")); // U+FF01 FULLWIDTH EXCLAMATION MARK
  check(SV("*\uff60*"), SV("{:*^4}"), SV("\uff60")); // FULLWIDTH RIGHT WHITE PARENTHESIS

  check(SV("*\uffe0*"), SV("{:*^4}"), SV("\uffe0")); // FULLWIDTH CENT SIGN
  check(SV("*\uffe6*"), SV("{:*^4}"), SV("\uffe6")); // FULLWIDTH WON SIGN

  //*** 4-byte code points ***
  check(SV("*\U00010000*"), SV("{:*^3}"), SV("\U00010000")); // LINEAR B SYLLABLE B008 A
  check(SV("*\U0010FFFF*"), SV("{:*^3}"), SV("\U0010FFFF")); // Undefined Character

  // 2 column ranges
  check(SV("*\U0001f300*"), SV("{:*^4}"), SV("\U0001f300")); // CYCLONE
  check(SV("*\U0001f64f*"), SV("{:*^4}"), SV("\U0001f64f")); // PERSON WITH FOLDED HANDS
  check(SV("*\U0001f900*"), SV("{:*^4}"), SV("\U0001f900")); // CIRCLED CROSS FORMEE WITH FOUR DOTS
  check(SV("*\U0001f9ff*"), SV("{:*^4}"), SV("\U0001f9ff")); // NAZAR AMULET
  check(SV("*\U00020000*"), SV("{:*^4}"), SV("\U00020000")); // <CJK Ideograph Extension B, First>
  check(SV("*\U0002fffd*"), SV("{:*^4}"), SV("\U0002fffd")); // Undefined Character
  check(SV("*\U00030000*"), SV("{:*^4}"), SV("\U00030000")); // <CJK Ideograph Extension G, First>
  check(SV("*\U0003fffd*"), SV("{:*^4}"), SV("\U0003fffd")); // Undefined Character
}

// One column output is unaffected.
// Two column output is removed, thus the result is only the fill character.
template <class CharT>
static void test_single_code_point_truncate() {
  //*** 1-byte code points ***
  check(SV("* *"), SV("{:*^3.1}"), SV(" "));
  check(SV("*~*"), SV("{:*^3.1}"), SV("~"));

  //*** 2-byte code points ***
  check(SV("*\u00a1*"), SV("{:*^3.1}"), SV("\u00a1")); // INVERTED EXCLAMATION MARK
  check(SV("*\u07ff*"), SV("{:*^3.1}"), SV("\u07ff")); // NKO TAMAN SIGN

  //*** 3.1-byte code points ***
  check(SV("*\u0800*"), SV("{:*^3.1}"), SV("\u0800")); // SAMARITAN LETTER ALAF
  check(SV("*\ufffd*"), SV("{:*^3.1}"), SV("\ufffd")); // REPLACEMENT CHARACTER

  // 2 column ranges
  check(SV("***"), SV("{:*^3.1}"), SV("\u1100")); // HANGUL CHOSEONG KIYEOK
  check(SV("***"), SV("{:*^3.1}"), SV("\u115f")); // HANGUL CHOSEONG FILLER

  check(SV("***"), SV("{:*^3.1}"), SV("\u2329")); // LEFT-POINTING ANGLE BRACKET
  check(SV("***"), SV("{:*^3.1}"), SV("\u232a")); // RIGHT-POINTING ANGLE BRACKET

  check(SV("***"), SV("{:*^3.1}"), SV("\u2e80")); // CJK RADICAL REPEAT
  check(SV("***"), SV("{:*^3.1}"), SV("\u303e")); // IDEOGRAPHIC VARIATION INDICATOR

  check(SV("***"), SV("{:*^3.1}"), SV("\u3040")); // U+3041 HIRAGANA LETTER SMALL A
  check(SV("***"), SV("{:*^3.1}"), SV("\ua4cf")); // U+A4D0 LISU LETTER BA

  check(SV("***"), SV("{:*^3.1}"), SV("\uac00")); // <Hangul Syllable, First>
  check(SV("***"), SV("{:*^3.1}"), SV("\ud7a3")); // Hangul Syllable Hih

  check(SV("***"), SV("{:*^3.1}"), SV("\uf900")); // CJK COMPATIBILITY IDEOGRAPH-F900
  check(SV("***"), SV("{:*^3.1}"), SV("\ufaff")); // U+FB00 LATIN SMALL LIGATURE FF

  check(SV("***"), SV("{:*^3.1}"), SV("\ufe10")); // PRESENTATION FORM FOR VERTICAL COMMA
  check(SV("***"), SV("{:*^3.1}"), SV("\ufe19")); // PRESENTATION FORM FOR VERTICAL HORIZONTAL ELLIPSIS

  check(SV("***"), SV("{:*^3.1}"), SV("\ufe30")); // PRESENTATION FORM FOR VERTICAL TWO DOT LEADER
  check(SV("***"), SV("{:*^3.1}"), SV("\ufe6f")); // U+FE70 ARABIC FATHATAN ISOLATED FORM

  check(SV("***"), SV("{:*^3.1}"), SV("\uff00")); // U+FF01 FULLWIDTH EXCLAMATION MARK
  check(SV("***"), SV("{:*^3.1}"), SV("\uff60")); // FULLWIDTH RIGHT WHITE PARENTHESIS

  check(SV("***"), SV("{:*^3.1}"), SV("\uffe0")); // FULLWIDTH CENT SIGN
  check(SV("***"), SV("{:*^3.1}"), SV("\uffe6")); // FULLWIDTH WON SIGN

  //*** 3.1-byte code points ***
  check(SV("*\U00010000*"), SV("{:*^3.1}"), SV("\U00010000")); // LINEAR B SYLLABLE B008 A
  check(SV("*\U0010FFFF*"), SV("{:*^3.1}"), SV("\U0010FFFF")); // Undefined Character

  // 2 column ranges
  check(SV("***"), SV("{:*^3.1}"), SV("\U0001f300")); // CYCLONE
  check(SV("***"), SV("{:*^3.1}"), SV("\U0001f64f")); // PERSON WITH FOLDED HANDS
  check(SV("***"), SV("{:*^3.1}"), SV("\U0001f900")); // CIRCLED CROSS FORMEE WITH FOUR DOTS
  check(SV("***"), SV("{:*^3.1}"), SV("\U0001f9ff")); // NAZAR AMULET
  check(SV("***"), SV("{:*^3.1}"), SV("\U00020000")); // <CJK Ideograph Extension B, First>
  check(SV("***"), SV("{:*^3.1}"), SV("\U0002fffd")); // Undefined Character
  check(SV("***"), SV("{:*^3.1}"), SV("\U00030000")); // <CJK Ideograph Extension G, First>
  check(SV("***"), SV("{:*^3.1}"), SV("\U0003fffd")); // Undefined Character
}

// The examples used in that paper.
template <class CharT>
static void test_P1868() {
  // Fill
  check(SV("*\u0041*"), SV("{:*^3}"), SV("\u0041")); // { LATIN CAPITAL LETTER A }
  check(SV("*\u00c1*"), SV("{:*^3}"), SV("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check(SV("*\u0041\u0301*"),
        SV("{:*^3}"),
        SV("\u0041\u0301"));                         // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check(SV("*\u0132*"), SV("{:*^3}"), SV("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check(SV("*\u0394*"), SV("{:*^3}"), SV("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check(SV("*\u0429*"), SV("{:*^3}"), SV("\u0429"));         // { CYRILLIC CAPITAL LETTER SHCHA }
  check(SV("*\u05d0*"), SV("{:*^3}"), SV("\u05d0"));         // { HEBREW LETTER ALEF }
  check(SV("*\u0634*"), SV("{:*^3}"), SV("\u0634"));         // { ARABIC LETTER SHEEN }
  check(SV("*\u3009*"), SV("{:*^4}"), SV("\u3009"));         // { RIGHT-POINTING ANGLE BRACKET }
  check(SV("*\u754c*"), SV("{:*^4}"), SV("\u754c"));         // { CJK Unified Ideograph-754C }
  check(SV("*\U0001f921*"), SV("{:*^4}"), SV("\U0001f921")); // { UNICORN FACE }
  check(SV("*\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466*"),
        SV("{:*^4}"),
        SV("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }

  // Truncate to 1 column: 1 column grapheme clusters are kept together.
  check(SV("*\u0041*"), SV("{:*^3.1}"), SV("\u0041")); // { LATIN CAPITAL LETTER A }
  check(SV("*\u00c1*"), SV("{:*^3.1}"), SV("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check(SV("*\u0041\u0301*"),
        SV("{:*^3.1}"),
        SV("\u0041\u0301"));                           // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check(SV("*\u0132*"), SV("{:*^3.1}"), SV("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check(SV("*\u0394*"), SV("{:*^3.1}"), SV("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check(SV("*\u0429*"), SV("{:*^3.1}"), SV("\u0429")); // { CYRILLIC CAPITAL LETTER SHCHA }
  check(SV("*\u05d0*"), SV("{:*^3.1}"), SV("\u05d0")); // { HEBREW LETTER ALEF }
  check(SV("*\u0634*"), SV("{:*^3.1}"), SV("\u0634")); // { ARABIC LETTER SHEEN }
  check(SV("***"), SV("{:*^3.1}"), SV("\u3009"));      // { RIGHT-POINTING ANGLE BRACKET }
  check(SV("***"), SV("{:*^3.1}"), SV("\u754c"));      // { CJK Unified Ideograph-754C }
  check(SV("***"), SV("{:*^3.1}"), SV("\U0001f921"));  // { UNICORN FACE }
  check(SV("***"),
        SV("{:*^3.1}"),
        SV("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }

  // Truncate to 2 column: 2 column grapheme clusters are kept together.
  check(SV("*\u0041*"), SV("{:*^3.2}"), SV("\u0041")); // { LATIN CAPITAL LETTER A }
  check(SV("*\u00c1*"), SV("{:*^3.2}"), SV("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check(SV("*\u0041\u0301*"),
        SV("{:*^3.2}"),
        SV("\u0041\u0301"));                           // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check(SV("*\u0132*"), SV("{:*^3.2}"), SV("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check(SV("*\u0394*"), SV("{:*^3.2}"), SV("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check(SV("*\u0429*"), SV("{:*^3.2}"), SV("\u0429"));         // { CYRILLIC CAPITAL LETTER SHCHA }
  check(SV("*\u05d0*"), SV("{:*^3.2}"), SV("\u05d0"));         // { HEBREW LETTER ALEF }
  check(SV("*\u0634*"), SV("{:*^3.2}"), SV("\u0634"));         // { ARABIC LETTER SHEEN }
  check(SV("*\u3009*"), SV("{:*^4.2}"), SV("\u3009"));         // { RIGHT-POINTING ANGLE BRACKET }
  check(SV("*\u754c*"), SV("{:*^4.2}"), SV("\u754c"));         // { CJK Unified Ideograph-754C }
  check(SV("*\U0001f921*"), SV("{:*^4.2}"), SV("\U0001f921")); // { UNICORN FACE }
  check(SV("*\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466*"),
        SV("{:*^4.2}"),
        SV("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }
}

#ifdef _LIBCPP_VERSION
// Tests the libc++ specific behaviour for malformed UTF-sequences. The
// Standard doesn't specify how to handle this.
template <class CharT>
static void test_malformed_code_point() {
  if constexpr (sizeof(CharT) == 1) {
    // Malformed at end.
    check(SV("*ZZZZ\x8f*"), SV("{:*^7}"), SV("ZZZZ\x8f"));
    check(SV("*ZZZZ\xcf*"), SV("{:*^7}"), SV("ZZZZ\xcf"));
    check(SV("*ZZZZ\xef*"), SV("{:*^7}"), SV("ZZZZ\xef"));
    check(SV("*ZZZZ\xff*"), SV("{:*^7}"), SV("ZZZZ\xff"));

    // Malformed in middle, no continuation
    check(SV("*ZZZZ\x8fZ*"), SV("{:*^8}"), SV("ZZZZ\x8fZ"));
    check(SV("*ZZZZ\xcfZ*"), SV("{:*^8}"), SV("ZZZZ\xcfZ"));
    check(SV("*ZZZZ\xefZ*"), SV("{:*^8}"), SV("ZZZZ\xefZ"));
    check(SV("*ZZZZ\xffZ*"), SV("{:*^8}"), SV("ZZZZ\xffZ"));

    check(SV("*ZZZZ\x8fZZ*"), SV("{:*^9}"), SV("ZZZZ\x8fZZ"));
    check(SV("*ZZZZ\xcfZZ*"), SV("{:*^9}"), SV("ZZZZ\xcfZZ"));
    check(SV("*ZZZZ\xefZZ*"), SV("{:*^9}"), SV("ZZZZ\xefZZ"));
    check(SV("*ZZZZ\xffZZ*"), SV("{:*^9}"), SV("ZZZZ\xffZZ"));

    check(SV("*ZZZZ\x8fZZZ*"), SV("{:*^10}"), SV("ZZZZ\x8fZZZ"));
    check(SV("*ZZZZ\xcfZZZ*"), SV("{:*^10}"), SV("ZZZZ\xcfZZZ"));
    check(SV("*ZZZZ\xefZZZ*"), SV("{:*^10}"), SV("ZZZZ\xefZZZ"));
    check(SV("*ZZZZ\xffZZZ*"), SV("{:*^10}"), SV("ZZZZ\xffZZZ"));

    check(SV("*ZZZZ\x8fZZZZ*"), SV("{:*^11}"), SV("ZZZZ\x8fZZZZ"));
    check(SV("*ZZZZ\xcfZZZZ*"), SV("{:*^11}"), SV("ZZZZ\xcfZZZZ"));
    check(SV("*ZZZZ\xefZZZZ*"), SV("{:*^11}"), SV("ZZZZ\xefZZZZ"));
    check(SV("*ZZZZ\xffZZZZ*"), SV("{:*^11}"), SV("ZZZZ\xffZZZZ"));

    // Premature end.
    check(SV("*ZZZZ\xef\xf5*"), SV("{:*^8}"), SV("ZZZZ\xef\xf5"));
    check(SV("*ZZZZ\xef\xf5ZZZZ*"), SV("{:*^12}"), SV("ZZZZ\xef\xf5ZZZZ"));
    check(SV("*ZZZZ\xff\xf5\xf5*"), SV("{:*^9}"), SV("ZZZZ\xff\xf5\xf5"));
    check(SV("*ZZZZ\xff\xf5\xf5ZZZZ*"), SV("{:*^13}"), SV("ZZZZ\xff\xf5\xf5ZZZZ"));

  } else if constexpr (sizeof(CharT) == 2) {
    // TODO FMT Add these tests.
  }
  // UTF-32 doesn't combine characters, thus no corruption tests.
}
#endif

template <class CharT>
static void test() {
  test_single_code_point_fill<CharT>();
  test_single_code_point_truncate<CharT>();
  test_P1868<CharT>();

#ifdef _LIBCPP_VERSION
  test_malformed_code_point<CharT>();
#endif
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
