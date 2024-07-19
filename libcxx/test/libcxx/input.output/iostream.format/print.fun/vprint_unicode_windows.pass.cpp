//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: no-wide-characters
// UNSUPPORTED: libcpp-has-no-unicode
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// Clang modules do not work with the definiton of _LIBCPP_TESTING_PRINT_WRITE_TO_WINDOWS_CONSOLE_FUNCTION
// XFAIL: clang-modules-build

// XFAIL: availability-fp_to_chars-missing

// <print>

// Tests the implementation of
//   void __print::__vprint_unicode_windows(FILE* __stream, string_view __fmt,
//                                          format_args __args, bool __write_nl,
//                                          bool __is_terminal);
//
// In the library when the stdout is redirected to a file it is no
// longer considered a terminal and the special terminal handling is no
// longer executed. By testing this function we can "force" emulate a
// terminal.
// Note __write_nl is tested by the public API.

#include <string_view>
#include <cstdio>
#include <algorithm>
#include <cassert>

void write_to_console(FILE*, std::wstring_view data);
#define _LIBCPP_TESTING_PRINT_WRITE_TO_WINDOWS_CONSOLE_FUNCTION ::write_to_console
#include <print>

#include "test_macros.h"
#include "filesystem_test_helper.h"
#include "make_string.h"

TEST_GCC_DIAGNOSTIC_IGNORED("-Wuse-after-free")

#define SV(S) MAKE_STRING_VIEW(wchar_t, S)

bool calling               = false;
std::wstring_view expected = L" world";

void write_to_console(FILE*, std::wstring_view data) {
  assert(calling);
  assert(data == expected);
}

scoped_test_env env;
std::string filename = env.create_file("output.txt");

static void test_basics() {
  FILE* file = std::fopen(filename.c_str(), "wb");
  assert(file);

  // Test writing to a "non-terminal" stream does not call WriteConsoleW.
  std::__print::__vprint_unicode_windows(file, "Hello", std::make_format_args(), false, false);
  assert(std::ftell(file) == 5);

  // It's not possible to reliably test whether writing to a "terminal" stream
  // flushes before writing. Testing flushing a closed stream worked on some
  // platforms, but was unreliable.
  calling = true;
  std::__print::__vprint_unicode_windows(file, " world", std::make_format_args(), false, true);
}

// When the output is a file the data is written as-is.
// When the output is a "terminal" invalid UTF-8 input is flagged.
static void test(std::wstring_view output, std::string_view input) {
  // *** File ***
  FILE* file = std::fopen(filename.c_str(), "wb");
  assert(file);

  std::__print::__vprint_unicode_windows(file, input, std::make_format_args(), false, false);
  assert(std::ftell(file) == static_cast<long>(input.size()));
  std::fclose(file);

  file = std::fopen(filename.c_str(), "rb");
  assert(file);

  std::vector<char> buffer(input.size());
  size_t read = fread(buffer.data(), 1, buffer.size(), file);
  assert(read == input.size());
  assert(input == std::string_view(buffer.begin(), buffer.end()));
  std::fclose(file);

  // *** Terminal ***
  expected = output;
  std::__print::__vprint_unicode_windows(file, input, std::make_format_args(), false, true);
}

static void test() {
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

  // *** Test invalid utf-8 ***
  test(SV("\ufffd"), "\xc3");
  test(SV("\ufffd("), "\xc3\x28");

  // surrogate range
  test(SV("\ufffd"), "\xed\xa0\x80"); // U+D800
  test(SV("\ufffd"), "\xed\xaf\xbf"); // U+DBFF
  test(SV("\ufffd"), "\xed\xbf\x80"); // U+DC00
  test(SV("\ufffd"), "\xed\xbf\xbf"); // U+DFFF

  // beyond valid values
  test(SV("\ufffd"), "\xf4\x90\x80\x80"); // U+110000
  test(SV("\ufffd"), "\xf4\xbf\xbf\xbf"); // U+11FFFF

  // Validates  http://unicode.org/review/pr-121.html option 3.
  test(SV("\u0061\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\u0062"), "\x61\xf1\x80\x80\xe1\x80\xc2\x62");
}

int main(int, char**) {
  test_basics();
  test();
}
