//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// Tests the Unicode width support of the standard format specifiers.
// It tests [format.string.std]/8 - 11:
// - Properly determining the estimated with of a unicode string.
// - Properly truncating to the wanted maximum width.

// This version runs the test when the platform has Unicode support.
// UNSUPPORTED: libcpp-has-no-unicode

#include <format>
#include <cassert>

#include "test_macros.h"
#include "make_string.h"

#define CSTR(S) MAKE_CSTRING(CharT, S)

using namespace std::__format_spec;

template <class CharT>
constexpr bool operator==(const __string_alignment<CharT>& lhs,
                          const __string_alignment<CharT>& rhs) noexcept {
  return lhs.__last == rhs.__last && lhs.__size == rhs.__size &&
         lhs.__align == rhs.__align;
}

template <class CharT>
constexpr void get_string_alignment(size_t offset, ptrdiff_t size, bool align,
                                    const CharT* str, size_t width,
                                    size_t precision) {
  std::basic_string_view<CharT> sv{str};
  __string_alignment<CharT> expected{sv.begin() + offset, size, align};
  __string_alignment<CharT> traits =
      __get_string_alignment(sv.begin(), sv.end(), width, precision);
  assert(traits == expected);
}

template <class CharT>
constexpr void estimate_column_width_fast(size_t expected, const CharT* str) {
  std::basic_string_view<CharT> sv{str};
  const CharT* out =
      __detail::__estimate_column_width_fast(sv.begin(), sv.end());
  assert(out == sv.begin() + expected);
}

template <class CharT>
constexpr void estimate_column_width_fast() {

  // No unicode
  estimate_column_width_fast(3, CSTR("abc"));
  estimate_column_width_fast(3, CSTR("a\u007fc"));

  if constexpr (sizeof(CharT) == 1) {
    // UTF-8 stop at the first multi-byte character.
    estimate_column_width_fast(0, CSTR("\u0080bc"));
    estimate_column_width_fast(1, CSTR("a\u0080c"));
    estimate_column_width_fast(2, CSTR("ab\u0080"));
    estimate_column_width_fast(1, CSTR("aßc"));

    estimate_column_width_fast(1, CSTR("a\u07ffc"));
    estimate_column_width_fast(1, CSTR("a\u0800c"));

    estimate_column_width_fast(1, CSTR("a\u10ffc"));
  } else {
    // UTF-16/32 stop at the first multi-column character.
    estimate_column_width_fast(3, CSTR("\u0080bc"));
    estimate_column_width_fast(3, CSTR("a\u0080c"));
    estimate_column_width_fast(3, CSTR("ab\u0080"));
    estimate_column_width_fast(3, CSTR("aßc"));

    estimate_column_width_fast(3, CSTR("a\u07ffc"));
    estimate_column_width_fast(3, CSTR("a\u0800c"));

    estimate_column_width_fast(3, CSTR("a\u10ffc"));
  }
  // First 2-column character
  estimate_column_width_fast(1, CSTR("a\u1100c"));

  estimate_column_width_fast(1, CSTR("a\U0000ffffc"));
  estimate_column_width_fast(1, CSTR("a\U00010000c"));
  estimate_column_width_fast(1, CSTR("a\U0010FFFFc"));
}

template <class CharT>
constexpr void estimate_column_width(size_t expected, const CharT* str) {
  std::basic_string_view<CharT> sv{str};
  std::__format_spec::__detail::__column_width_result<CharT> column_info =
      __detail::__estimate_column_width(sv.begin(), sv.end(), -1);
  assert(column_info.__width == expected);
}

template <class CharT>
constexpr void estimate_column_width() {
  //*** 1-byte code points ***
  estimate_column_width(1, CSTR(" "));
  estimate_column_width(1, CSTR("~"));

  //*** 2-byte code points ***
  estimate_column_width(1, CSTR("\u00a1")); // INVERTED EXCLAMATION MARK
  estimate_column_width(1, CSTR("\u07ff")); // NKO TAMAN SIGN

  //*** 3-byte code points ***
  estimate_column_width(1, CSTR("\u0800")); // SAMARITAN LETTER ALAF
  estimate_column_width(1, CSTR("\ufffd")); // REPLACEMENT CHARACTER

  // 2 column ranges
  estimate_column_width(2, CSTR("\u1100")); // HANGUL CHOSEONG KIYEOK
  estimate_column_width(2, CSTR("\u115f")); // HANGUL CHOSEONG FILLER

  estimate_column_width(2, CSTR("\u2329")); // LEFT-POINTING ANGLE BRACKET
  estimate_column_width(2, CSTR("\u232a")); // RIGHT-POINTING ANGLE BRACKET

  estimate_column_width(2, CSTR("\u2e80")); // CJK RADICAL REPEAT
  estimate_column_width(2, CSTR("\u303e")); // IDEOGRAPHIC VARIATION INDICATOR

  estimate_column_width(2, CSTR("\u3040")); // U+3041 HIRAGANA LETTER SMALL A
  estimate_column_width(2, CSTR("\ua4cf")); // U+A4D0 LISU LETTER BA

  estimate_column_width(2, CSTR("\uac00")); // <Hangul Syllable, First>
  estimate_column_width(2, CSTR("\ud7a3")); // Hangul Syllable Hih

  estimate_column_width(2, CSTR("\uf900")); // CJK COMPATIBILITY IDEOGRAPH-F900
  estimate_column_width(2, CSTR("\ufaff")); // U+FB00 LATIN SMALL LIGATURE FF

  estimate_column_width(2,
                        CSTR("\ufe10")); // PRESENTATION FORM FOR VERTICAL COMMA
  estimate_column_width(
      2, CSTR("\ufe19")); // PRESENTATION FORM FOR VERTICAL HORIZONTAL ELLIPSIS

  estimate_column_width(
      2, CSTR("\ufe30")); // PRESENTATION FORM FOR VERTICAL TWO DOT LEADER
  estimate_column_width(2,
                        CSTR("\ufe6f")); // U+FE70 ARABIC FATHATAN ISOLATED FORM

  estimate_column_width(2, CSTR("\uff00")); // U+FF01 FULLWIDTH EXCLAMATION MARK
  estimate_column_width(2, CSTR("\uff60")); // FULLWIDTH RIGHT WHITE PARENTHESIS

  estimate_column_width(2, CSTR("\uffe0")); // FULLWIDTH CENT SIGN
  estimate_column_width(2, CSTR("\uffe6")); // FULLWIDTH WON SIGN

  //*** 4-byte code points ***
  estimate_column_width(1, CSTR("\U00010000")); // LINEAR B SYLLABLE B008 A
  estimate_column_width(1, CSTR("\U0010FFFF")); // Undefined Character

  // 2 column ranges
  estimate_column_width(2, CSTR("\U0001f300")); // CYCLONE
  estimate_column_width(2, CSTR("\U0001f64f")); // PERSON WITH FOLDED HANDS
  estimate_column_width(
      2, CSTR("\U0001f900")); // CIRCLED CROSS FORMEE WITH FOUR DOTS
  estimate_column_width(2, CSTR("\U0001f9ff")); // NAZAR AMULET
  estimate_column_width(
      2, CSTR("\U00020000")); // <CJK Ideograph Extension B, First>
  estimate_column_width(2, CSTR("\U0002fffd")); // Undefined Character
  estimate_column_width(
      2, CSTR("\U00030000")); // <CJK Ideograph Extension G, First>
  estimate_column_width(2, CSTR("\U0003fffd")); // Undefined Character
}

template <class CharT>
constexpr void get_string_alignment() {
  // Truncate the input.
  get_string_alignment(2, 2, false, CSTR("abc"), 0, 2);

  // The 2-column character gets entirely rejected.
  get_string_alignment(1, 1, false, CSTR("a\u115f"), 0, 2);

  // Due to the requested width extra alignment is required.
  get_string_alignment(1, 1, true, CSTR("a\u115f"), 2, 2);

  // Same but for a 2-column 4-byte UTF-8 sequence
  get_string_alignment(1, 1, false, CSTR("a\U0001f300"), 0, 2);
  get_string_alignment(1, 1, true, CSTR("a\U0001f300"), 2, 2);

  // No alignment required.
  get_string_alignment(3, 3, false, CSTR("abc"), 2, -1);
  get_string_alignment(3, 3, false, CSTR("abc"), 3, -1);

  // Special case, we have a special character already parsed and have enough
  // withd to satisfy the minumum required width.
  get_string_alignment(3 + 2 * (sizeof(CharT) == 1), 0, false, CSTR("ab\u1111"),
                       2, -1);

  // Evaluates all so size ->4
  get_string_alignment(3 + 2 * (sizeof(CharT) == 1), 4, false,
                       CSTR("a\u115fc") /* 2-column character */, 3, -1);
  // Evaluates all so size ->4
  get_string_alignment(3 + 2 * (sizeof(CharT) == 1), 4, false,
                       CSTR("a\u115fc") /* 2-column character */, 4, -1);

  // Evaluates all so size ->5
  get_string_alignment(4 + 2 * (sizeof(CharT) == 1), 5, false,
                       CSTR("a\u115fcd") /* 2-column character */, 4, -1);

  // Evaluates all so size ->5
  get_string_alignment(4 + 2 * (sizeof(CharT) == 1), 5, false,
                       CSTR("a\u115fcd") /* 2-column character */, 5, -1);

  // Extend width
  get_string_alignment(3, 3, true, CSTR("abc"), 4, -1);
  get_string_alignment(3 + 2 * (sizeof(CharT) == 1), 3, true,
                       CSTR("a\u1160c") /* 1-column character */, 4, -1);

  // In this case the threshold where the width is still determined.
  get_string_alignment(2 + 2 * (sizeof(CharT) == 1), 3, false, CSTR("i\u1110"),
                       2, -1);

  // The width is no longer exactly determined.
  get_string_alignment(2 + 2 * (sizeof(CharT) == 1), 0, false, CSTR("i\u1110"),
                       1, -1);

  // Extend width and truncate input.
  get_string_alignment(1, 1, true, CSTR("abc"), 3, 1);

  if constexpr (sizeof(CharT) == 1) {
    // Corrupt UTF-8 sequence.
    get_string_alignment(2, 2, false, CSTR("a\xc0"), 0, 3);
    get_string_alignment(2, 2, false, CSTR("a\xe0"), 0, 3);
    get_string_alignment(2, 2, false, CSTR("a\xf0"), 0, 3);
  } else if constexpr (sizeof(CharT) == 2) {
    // Corrupt UTF-16 sequence.
    if constexpr (std::same_as<CharT, char16_t>)
      get_string_alignment(2, 2, false, u"a\xdddd", 0, 3);
    else
      // Corrupt UTF-16 wchar_t seqence.
      get_string_alignment(2, 2, false, L"a\xdddd", 0, 3);
  }
  // UTF-32 doesn't combine characters, thus no corruption tests.
}

template <class CharT>
constexpr void test() {
  estimate_column_width_fast<CharT>();
  estimate_column_width<CharT>();
  get_string_alignment<CharT>();
}

constexpr bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
#ifndef _LIBCPP_HAS_NO_CHAR8_T
  test<char8_t>();
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
  test<char16_t>();
  test<char32_t>();
#endif
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
