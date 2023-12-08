//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <print>

// Tests the UTF-8 to UTF-16/32 encoding.
// UTF-16 is used on Windows to write to the Unicode API.
// UTF-32 is used to test the Windows behaviour on Linux using 32-bit wchar_t.

#include <algorithm>
#include <array>
#include <cassert>
#include <print>
#include <string_view>

#include "test_macros.h"
#include "make_string.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
constexpr void test(std::basic_string_view<CharT> expected, std::string_view input) {
  assert(expected.size() < 1024);
  std::array<CharT, 1024> buffer;
  std::ranges::fill(buffer, CharT('*'));

  CharT* out = std::__unicode::__transcode(input.begin(), input.end(), buffer.data());

  assert(std::basic_string_view<CharT>(buffer.data(), out) == expected);

  out = std::find_if(out, buffer.end(), [](CharT c) { return c != CharT('*'); });
  assert(out == buffer.end());
}

template <class CharT>
constexpr void test() {
  // *** Test valid UTF-8 ***
#define TEST(S) test(SV(S), S)
  TEST("hello world");
  // copied from benchmarks/std_format_spec_string_unicode.bench.cpp
  TEST("Lorem ipsum dolor sit amet, ne sensibus evertitur aliquando his. Iuvaret fabulas qui ex.");
  TEST("Lōrem ipsūm dolor sīt æmeÞ, ea vel nostrud feuġǣit, muciūs tēmporiȝusrefērrēnÞur no mel.");
  TEST("Лорем ипсум долор сит амет, еу диам тамяуам принципес вис, еяуидем цонцептам диспутандо");
  TEST("入ト年媛ろ舗学ラロ準募ケカ社金スノ屋検れう策他セヲシ引口ぎ集7独ぱクふ出車ぽでぱ円輪ルノ受打わ。");
  TEST("\U0001f636\u200d\U0001f32b\ufe0f");
#undef TEST

  // *** Test invalid UTF-8 ***
  test(SV("\ufffd"), "\xc3");
  test(SV("\ufffd("), "\xc3\x28");

  // Surrogate range
  test(SV("\ufffd"), "\xed\xa0\x80"); // U+D800
  test(SV("\ufffd"), "\xed\xaf\xbf"); // U+DBFF
  test(SV("\ufffd"), "\xed\xbf\x80"); // U+DC00
  test(SV("\ufffd"), "\xed\xbf\xbf"); // U+DFFF

  // Beyond valid values
  test(SV("\ufffd"), "\xf4\x90\x80\x80"); // U+110000
  test(SV("\ufffd"), "\xf4\xbf\xbf\xbf"); // U+11FFFF

  // Validates http://unicode.org/review/pr-121.html option 3.
  test(SV("\u0061\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\u0062"), "\x61\xF1\x80\x80\xE1\x80\xC2\x62");
}

constexpr bool test() {
  test<char16_t>();
  test<char32_t>();
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  test<wchar_t>();
#endif
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
