//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// UTF-32 doesn't work properly
// XFAIL: windows

// <format>

// Tests the Unicode width support of the standard format specifiers.
// It tests [format.string.std]/8 - 11:
// - Properly determining the estimated with of a unicode string.
// - Properly truncating to the wanted maximum width.

// This version runs the test when the platform doesn't have Unicode support.
// REQUIRES: libcpp-has-no-unicode

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
constexpr void get_string_alignment() {
  // Truncate the input.
  get_string_alignment(2, 2, false, CSTR("abc"), 0, 2);

  // The 2-column character gets half accepted.
  get_string_alignment(2, 2, false, CSTR("a\u115f"), 0, 2);

  // No alignment since the number of characters fits.
  get_string_alignment(2, 2, false, CSTR("a\u115f"), 2, 2);

  // Same but for a 2-column 4-byte UTF-8 sequence
  get_string_alignment(2, 2, false, CSTR("a\U0001f300"), 0, 2);
  get_string_alignment(2, 2, false, CSTR("a\U0001f300"), 2, 2);

  // No alignment required.
  get_string_alignment(3, 3, false, CSTR("abc"), 2, -1);
  get_string_alignment(3, 3, false, CSTR("abc"), 3, -1);

  get_string_alignment(3 + 2 * (sizeof(CharT) == 1),
                       3 + 2 * (sizeof(CharT) == 1), false, CSTR("ab\u1111"), 2,
                       -1);

  // Doesn't evaluate 'c' so size -> 0
  get_string_alignment(3 + 2 * (sizeof(CharT) == 1),
                       3 + 2 * (sizeof(CharT) == 1), false,
                       CSTR("a\u115fc") /* 2-column character */, 3, -1);
  // Extend width
  get_string_alignment(3, 3, true, CSTR("abc"), 4, -1);
  get_string_alignment(3 + 2 * (sizeof(CharT) == 1),
                       3 + 2 * (sizeof(CharT) == 1), true,
                       CSTR("a\u1160c") /* 1-column character */, 6, -1);
}

template <class CharT>
constexpr void test() {
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
