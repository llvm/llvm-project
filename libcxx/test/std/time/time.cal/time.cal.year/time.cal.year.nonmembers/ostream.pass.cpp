//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
// class year;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year& year);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

#define TEST_EQUAL(OUT, EXPECTED)                                                                                      \
  TEST_REQUIRE(OUT == EXPECTED,                                                                                        \
               TEST_WRITE_CONCATENATED(                                                                                \
                   "\nExpression      ", #OUT, "\nExpected output ", EXPECTED, "\nActual output   ", OUT, '\n'));

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::year year) {
  std::basic_stringstream<CharT> sstr;
  sstr << year;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::year year) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << year;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::year year) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << year;
  return sstr.str();
}

template <class CharT>
static void test() {
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year{-32'768}), SV("-32768 is not a valid year"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year{-32'767}), SV("-32767"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year{0}), SV("0000"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year{1970}), SV("1970"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year{32'767}), SV("32767"));

  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year{-32'768}), SV("-32768 is not a valid year"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year{-32'767}), SV("-32767"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year{0}), SV("0000"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year{1970}), SV("1970"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year{32'767}), SV("32767"));

  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year{-32'768}), SV("-32768 is not a valid year"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year{-32'767}), SV("-32767"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year{0}), SV("0000"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year{1970}), SV("1970"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year{32'767}), SV("32767"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
