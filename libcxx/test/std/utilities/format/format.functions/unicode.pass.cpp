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

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <iostream>
#  include <type_traits>
#endif

#define SV(S) MAKE_STRING_VIEW(CharT, S)

auto check = []<string_literal fmt, class CharT, class... Args>(
    std::basic_string_view<CharT> expected, const Args&... args) constexpr {
  std::basic_string<CharT> out = std::format(fmt.template sv<CharT>(), args...);
#ifndef TEST_HAS_NO_LOCALIZATION
  if constexpr (std::same_as<CharT, char>)
    if (out != expected)
      std::cerr << "\nFormat string   " << fmt.template sv<char>() << "\nExpected output " << expected
                << "\nActual output   " << out << '\n';
#endif
  assert(out == expected);
};

template <class CharT>
static void test_single_code_point_fill() {
  //*** 1-byte code points ***
  check.template operator()<"{:*^3}">(SV("* *"), SV(" "));
  check.template operator()<"{:*^3}">(SV("*~*"), SV("~"));

  //*** 2-byte code points ***
  check.template operator()<"{:*^3}">(SV("*\u00a1*"), SV("\u00a1")); // INVERTED EXCLAMATION MARK
  check.template operator()<"{:*^3}">(SV("*\u07ff*"), SV("\u07ff")); // NKO TAMAN SIGN

  //*** 3-byte code points ***
  check.template operator()<"{:*^3}">(SV("*\u0800*"), SV("\u0800")); // SAMARITAN LETTER ALAF
  check.template operator()<"{:*^3}">(SV("*\ufffd*"), SV("\ufffd")); // REPLACEMENT CHARACTER

  // 2 column ranges
  check.template operator()<"{:*^4}">(SV("*\u1100*"), SV("\u1100")); // HANGUL CHOSEONG KIYEOK
  check.template operator()<"{:*^4}">(SV("*\u115f*"), SV("\u115f")); // HANGUL CHOSEONG FILLER

  check.template operator()<"{:*^4}">(SV("*\u2329*"), SV("\u2329")); // LEFT-POINTING ANGLE BRACKET
  check.template operator()<"{:*^4}">(SV("*\u232a*"), SV("\u232a")); // RIGHT-POINTING ANGLE BRACKET

  check.template operator()<"{:*^4}">(SV("*\u2e80*"), SV("\u2e80")); // CJK RADICAL REPEAT
  check.template operator()<"{:*^4}">(SV("*\u303e*"), SV("\u303e")); // IDEOGRAPHIC VARIATION INDICATOR

  check.template operator()<"{:*^4}">(SV("*\u3040*"), SV("\u3040")); // U+3041 HIRAGANA LETTER SMALL A
  check.template operator()<"{:*^4}">(SV("*\ua4cf*"), SV("\ua4cf")); // U+A4D0 LISU LETTER BA

  check.template operator()<"{:*^4}">(SV("*\uac00*"), SV("\uac00")); // <Hangul Syllable, First>
  check.template operator()<"{:*^4}">(SV("*\ud7a3*"), SV("\ud7a3")); // Hangul Syllable Hih

  check.template operator()<"{:*^4}">(SV("*\uf900*"), SV("\uf900")); // CJK COMPATIBILITY IDEOGRAPH-F900
  check.template operator()<"{:*^4}">(SV("*\ufaff*"), SV("\ufaff")); // U+FB00 LATIN SMALL LIGATURE FF

  check.template operator()<"{:*^4}">(SV("*\ufe10*"), SV("\ufe10")); // PRESENTATION FORM FOR VERTICAL COMMA
  check.template
  operator()<"{:*^4}">(SV("*\ufe19*"), SV("\ufe19")); // PRESENTATION FORM FOR VERTICAL HORIZONTAL ELLIPSIS

  check.template operator()<"{:*^4}">(SV("*\ufe30*"), SV("\ufe30")); // PRESENTATION FORM FOR VERTICAL TWO DOT LEADER
  check.template operator()<"{:*^4}">(SV("*\ufe6f*"), SV("\ufe6f")); // U+FE70 ARABIC FATHATAN ISOLATED FORM

  check.template operator()<"{:*^4}">(SV("*\uff00*"), SV("\uff00")); // U+FF01 FULLWIDTH EXCLAMATION MARK
  check.template operator()<"{:*^4}">(SV("*\uff60*"), SV("\uff60")); // FULLWIDTH RIGHT WHITE PARENTHESIS

  check.template operator()<"{:*^4}">(SV("*\uffe0*"), SV("\uffe0")); // FULLWIDTH CENT SIGN
  check.template operator()<"{:*^4}">(SV("*\uffe6*"), SV("\uffe6")); // FULLWIDTH WON SIGN

  //*** 4-byte code points ***
  check.template operator()<"{:*^3}">(SV("*\U00010000*"), SV("\U00010000")); // LINEAR B SYLLABLE B008 A
  check.template operator()<"{:*^3}">(SV("*\U0010FFFF*"), SV("\U0010FFFF")); // Undefined Character

  // 2 column ranges
  check.template operator()<"{:*^4}">(SV("*\U0001f300*"), SV("\U0001f300")); // CYCLONE
  check.template operator()<"{:*^4}">(SV("*\U0001f64f*"), SV("\U0001f64f")); // PERSON WITH FOLDED HANDS
  check.template operator()<"{:*^4}">(SV("*\U0001f900*"), SV("\U0001f900")); // CIRCLED CROSS FORMEE WITH FOUR DOTS
  check.template operator()<"{:*^4}">(SV("*\U0001f9ff*"), SV("\U0001f9ff")); // NAZAR AMULET
  check.template operator()<"{:*^4}">(SV("*\U00020000*"), SV("\U00020000")); // <CJK Ideograph Extension B, First>
  check.template operator()<"{:*^4}">(SV("*\U0002fffd*"), SV("\U0002fffd")); // Undefined Character
  check.template operator()<"{:*^4}">(SV("*\U00030000*"), SV("\U00030000")); // <CJK Ideograph Extension G, First>
  check.template operator()<"{:*^4}">(SV("*\U0003fffd*"), SV("\U0003fffd")); // Undefined Character
}

// One column output is unaffected.
// Two column output is removed, thus the result is only the fill character.
template <class CharT>
static void test_single_code_point_truncate() {
  //*** 1-byte code points ***
  check.template operator()<"{:*^3.1}">(SV("* *"), SV(" "));
  check.template operator()<"{:*^3.1}">(SV("*~*"), SV("~"));

  //*** 2-byte code points ***
  check.template operator()<"{:*^3.1}">(SV("*\u00a1*"), SV("\u00a1")); // INVERTED EXCLAMATION MARK
  check.template operator()<"{:*^3.1}">(SV("*\u07ff*"), SV("\u07ff")); // NKO TAMAN SIGN

  //*** 3.1-byte code points ***
  check.template operator()<"{:*^3.1}">(SV("*\u0800*"), SV("\u0800")); // SAMARITAN LETTER ALAF
  check.template operator()<"{:*^3.1}">(SV("*\ufffd*"), SV("\ufffd")); // REPLACEMENT CHARACTER

  // 2 column ranges
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u1100")); // HANGUL CHOSEONG KIYEOK
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u115f")); // HANGUL CHOSEONG FILLER

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u2329")); // LEFT-POINTING ANGLE BRACKET
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u232a")); // RIGHT-POINTING ANGLE BRACKET

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u2e80")); // CJK RADICAL REPEAT
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u303e")); // IDEOGRAPHIC VARIATION INDICATOR

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u3040")); // U+3041 HIRAGANA LETTER SMALL A
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ua4cf")); // U+A4D0 LISU LETTER BA

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\uac00")); // <Hangul Syllable, First>
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ud7a3")); // Hangul Syllable Hih

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\uf900")); // CJK COMPATIBILITY IDEOGRAPH-F900
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ufaff")); // U+FB00 LATIN SMALL LIGATURE FF

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ufe10")); // PRESENTATION FORM FOR VERTICAL COMMA
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ufe19")); // PRESENTATION FORM FOR VERTICAL HORIZONTAL ELLIPSIS

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ufe30")); // PRESENTATION FORM FOR VERTICAL TWO DOT LEADER
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\ufe6f")); // U+FE70 ARABIC FATHATAN ISOLATED FORM

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\uff00")); // U+FF01 FULLWIDTH EXCLAMATION MARK
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\uff60")); // FULLWIDTH RIGHT WHITE PARENTHESIS

  check.template operator()<"{:*^3.1}">(SV("***"), SV("\uffe0")); // FULLWIDTH CENT SIGN
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\uffe6")); // FULLWIDTH WON SIGN

  //*** 3.1-byte code points ***
  check.template operator()<"{:*^3.1}">(SV("*\U00010000*"), SV("\U00010000")); // LINEAR B SYLLABLE B008 A
  check.template operator()<"{:*^3.1}">(SV("*\U0010FFFF*"), SV("\U0010FFFF")); // Undefined Character

  // 2 column ranges
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0001f300")); // CYCLONE
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0001f64f")); // PERSON WITH FOLDED HANDS
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0001f900")); // CIRCLED CROSS FORMEE WITH FOUR DOTS
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0001f9ff")); // NAZAR AMULET
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U00020000")); // <CJK Ideograph Extension B, First>
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0002fffd")); // Undefined Character
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U00030000")); // <CJK Ideograph Extension G, First>
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0003fffd")); // Undefined Character
}

// The examples used in that paper.
template <class CharT>
static void test_P1868() {
  // Fill
  check.template operator()<"{:*^3}">(SV("*\u0041*"), SV("\u0041")); // { LATIN CAPITAL LETTER A }
  check.template operator()<"{:*^3}">(SV("*\u00c1*"), SV("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check.template operator()<"{:*^3}">(
      SV("*\u0041\u0301*"),
      SV("\u0041\u0301")); // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check.template operator()<"{:*^3}">(SV("*\u0132*"), SV("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check.template operator()<"{:*^3}">(SV("*\u0394*"), SV("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check.template operator()<"{:*^3}">(SV("*\u0429*"), SV("\u0429"));         // { CYRILLIC CAPITAL LETTER SHCHA }
  check.template operator()<"{:*^3}">(SV("*\u05d0*"), SV("\u05d0"));         // { HEBREW LETTER ALEF }
  check.template operator()<"{:*^3}">(SV("*\u0634*"), SV("\u0634"));         // { ARABIC LETTER SHEEN }
  check.template operator()<"{:*^4}">(SV("*\u3009*"), SV("\u3009"));         // { RIGHT-POINTING ANGLE BRACKET }
  check.template operator()<"{:*^4}">(SV("*\u754c*"), SV("\u754c"));         // { CJK Unified Ideograph-754C }
  check.template operator()<"{:*^4}">(SV("*\U0001f921*"), SV("\U0001f921")); // { UNICORN FACE }
  check.template operator()<"{:*^4}">(
      SV("*\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466*"),
      SV("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }

  // Truncate to 1 column: 1 column grapheme clusters are kept together.
  check.template operator()<"{:*^3.1}">(SV("*\u0041*"), SV("\u0041")); // { LATIN CAPITAL LETTER A }
  check.template operator()<"{:*^3.1}">(SV("*\u00c1*"), SV("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check.template operator()<"{:*^3.1}">(
      SV("*\u0041\u0301*"),
      SV("\u0041\u0301")); // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check.template operator()<"{:*^3.1}">(SV("*\u0132*"), SV("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check.template operator()<"{:*^3.1}">(SV("*\u0394*"), SV("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check.template operator()<"{:*^3.1}">(SV("*\u0429*"), SV("\u0429")); // { CYRILLIC CAPITAL LETTER SHCHA }
  check.template operator()<"{:*^3.1}">(SV("*\u05d0*"), SV("\u05d0")); // { HEBREW LETTER ALEF }
  check.template operator()<"{:*^3.1}">(SV("*\u0634*"), SV("\u0634")); // { ARABIC LETTER SHEEN }
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u3009"));      // { RIGHT-POINTING ANGLE BRACKET }
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\u754c"));      // { CJK Unified Ideograph-754C }
  check.template operator()<"{:*^3.1}">(SV("***"), SV("\U0001f921"));  // { UNICORN FACE }
  check.template operator()<"{:*^3.1}">(
      SV("***"),
      SV("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }

  // Truncate to 2 column: 2 column grapheme clusters are kept together.
  check.template operator()<"{:*^3.2}">(SV("*\u0041*"), SV("\u0041")); // { LATIN CAPITAL LETTER A }
  check.template operator()<"{:*^3.2}">(SV("*\u00c1*"), SV("\u00c1")); // { LATIN CAPITAL LETTER A WITH ACUTE }
  check.template operator()<"{:*^3.2}">(
      SV("*\u0041\u0301*"),
      SV("\u0041\u0301")); // { LATIN CAPITAL LETTER A } { COMBINING ACUTE ACCENT }
  check.template operator()<"{:*^3.2}">(SV("*\u0132*"), SV("\u0132")); // { LATIN CAPITAL LIGATURE IJ }
  check.template operator()<"{:*^3.2}">(SV("*\u0394*"), SV("\u0394")); // { GREEK CAPITAL LETTER DELTA }

  check.template operator()<"{:*^3.2}">(SV("*\u0429*"), SV("\u0429"));         // { CYRILLIC CAPITAL LETTER SHCHA }
  check.template operator()<"{:*^3.2}">(SV("*\u05d0*"), SV("\u05d0"));         // { HEBREW LETTER ALEF }
  check.template operator()<"{:*^3.2}">(SV("*\u0634*"), SV("\u0634"));         // { ARABIC LETTER SHEEN }
  check.template operator()<"{:*^4.2}">(SV("*\u3009*"), SV("\u3009"));         // { RIGHT-POINTING ANGLE BRACKET }
  check.template operator()<"{:*^4.2}">(SV("*\u754c*"), SV("\u754c"));         // { CJK Unified Ideograph-754C }
  check.template operator()<"{:*^4.2}">(SV("*\U0001f921*"), SV("\U0001f921")); // { UNICORN FACE }
  check.template operator()<"{:*^4.2}">(
      SV("*\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466*"),
      SV("\U0001f468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466")); // { Family: Man, Woman, Girl, Boy }
}

#ifdef _LIBCPP_VERSION
// Tests the libc++ specific behaviour for malformed UTF-sequences. The
// Standard doesn't specify how to handle this.
template <class CharT>
static void test_malformed_code_point() {
  if constexpr (sizeof(CharT) == 1) {
    // Malformed at end.
    check.template operator()<"{:*^7}">(SV("*ZZZZ\x8f*"), SV("ZZZZ\x8f"));
    check.template operator()<"{:*^7}">(SV("*ZZZZ\xcf*"), SV("ZZZZ\xcf"));
    check.template operator()<"{:*^7}">(SV("*ZZZZ\xef*"), SV("ZZZZ\xef"));
    check.template operator()<"{:*^7}">(SV("*ZZZZ\xff*"), SV("ZZZZ\xff"));

    // Malformed in middle, no continuation
    check.template operator()<"{:*^8}">(SV("*ZZZZ\x8fZ*"), SV("ZZZZ\x8fZ"));
    check.template operator()<"{:*^8}">(SV("*ZZZZ\xcfZ*"), SV("ZZZZ\xcfZ"));
    check.template operator()<"{:*^8}">(SV("*ZZZZ\xefZ*"), SV("ZZZZ\xefZ"));
    check.template operator()<"{:*^8}">(SV("*ZZZZ\xffZ*"), SV("ZZZZ\xffZ"));

    check.template operator()<"{:*^9}">(SV("*ZZZZ\x8fZZ*"), SV("ZZZZ\x8fZZ"));
    check.template operator()<"{:*^9}">(SV("*ZZZZ\xcfZZ*"), SV("ZZZZ\xcfZZ"));
    check.template operator()<"{:*^9}">(SV("*ZZZZ\xefZZ*"), SV("ZZZZ\xefZZ"));
    check.template operator()<"{:*^9}">(SV("*ZZZZ\xffZZ*"), SV("ZZZZ\xffZZ"));

    check.template operator()<"{:*^10}">(SV("*ZZZZ\x8fZZZ*"), SV("ZZZZ\x8fZZZ"));
    check.template operator()<"{:*^10}">(SV("*ZZZZ\xcfZZZ*"), SV("ZZZZ\xcfZZZ"));
    check.template operator()<"{:*^10}">(SV("*ZZZZ\xefZZZ*"), SV("ZZZZ\xefZZZ"));
    check.template operator()<"{:*^10}">(SV("*ZZZZ\xffZZZ*"), SV("ZZZZ\xffZZZ"));

    check.template operator()<"{:*^11}">(SV("*ZZZZ\x8fZZZZ*"), SV("ZZZZ\x8fZZZZ"));
    check.template operator()<"{:*^11}">(SV("*ZZZZ\xcfZZZZ*"), SV("ZZZZ\xcfZZZZ"));
    check.template operator()<"{:*^11}">(SV("*ZZZZ\xefZZZZ*"), SV("ZZZZ\xefZZZZ"));
    check.template operator()<"{:*^11}">(SV("*ZZZZ\xffZZZZ*"), SV("ZZZZ\xffZZZZ"));

    // Premature end.
    check.template operator()<"{:*^8}">(SV("*ZZZZ\xef\xf5*"), SV("ZZZZ\xef\xf5"));
    check.template operator()<"{:*^12}">(SV("*ZZZZ\xef\xf5ZZZZ*"), SV("ZZZZ\xef\xf5ZZZZ"));
    check.template operator()<"{:*^9}">(SV("*ZZZZ\xff\xf5\xf5*"), SV("ZZZZ\xff\xf5\xf5"));
    check.template operator()<"{:*^13}">(SV("*ZZZZ\xff\xf5\xf5ZZZZ*"), SV("ZZZZ\xff\xf5\xf5ZZZZ"));

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
