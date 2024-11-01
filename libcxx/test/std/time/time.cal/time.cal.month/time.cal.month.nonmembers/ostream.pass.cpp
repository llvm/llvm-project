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

// class month;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month& month);

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
static std::basic_string<CharT> stream_c_locale(std::chrono::month month) {
  std::basic_stringstream<CharT> sstr;
  sstr << month;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month month) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << month;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month month) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << month;
  return sstr.str();
}

template <class CharT>
static void test() {
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{0}), SV("0 is not a valid month"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{1}), SV("Jan"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{2}), SV("Feb"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{3}), SV("Mar"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{4}), SV("Apr"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{5}), SV("May"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{6}), SV("Jun"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{7}), SV("Jul"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{8}), SV("Aug"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{9}), SV("Sep"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{10}), SV("Oct"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{11}), SV("Nov"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{12}), SV("Dec"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{13}), SV("13 is not a valid month"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month{255}), SV("255 is not a valid month"));

  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{0}), SV("0 is not a valid month"));
#if defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{1}), SV("jan"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{2}), SV("fév"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{3}), SV("mar"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{4}), SV("avr"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{5}), SV("mai"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{6}), SV("jui"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{7}), SV("jul"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{8}), SV("aoû"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{9}), SV("sep"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{10}), SV("oct"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{11}), SV("nov"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{12}), SV("déc"));
#else //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{1}), SV("janv."));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{2}), SV("févr."));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{3}), SV("mars"));
#  if defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{4}), SV("avr."));
#  else
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{4}), SV("avril"));
#  endif
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{5}), SV("mai"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{6}), SV("juin"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{7}), SV("juil."));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{8}), SV("août"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{9}), SV("sept."));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{10}), SV("oct."));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{11}), SV("nov."));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{12}), SV("déc."));
#endif //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{13}), SV("13 is not a valid month"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month{255}), SV("255 is not a valid month"));

  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{0}), SV("0 is not a valid month"));
#if defined(__APPLE__) || defined(_WIN32)
#  if defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{1}), SV(" 1"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{2}), SV(" 2"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{3}), SV(" 3"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{4}), SV(" 4"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{5}), SV(" 5"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{6}), SV(" 6"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{7}), SV(" 7"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{8}), SV(" 8"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{9}), SV(" 9"));
#  else  //  defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{1}), SV("1"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{2}), SV("2"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{3}), SV("3"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{4}), SV("4"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{5}), SV("5"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{6}), SV("6"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{7}), SV("7"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{8}), SV("8"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{9}), SV("9"));
#  endif // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{10}), SV("10"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{11}), SV("11"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{12}), SV("12"));
#else // defined(__APPLE__)|| defined(_WIN32)
#  if defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{1}), SV("1月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{2}), SV("2月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{3}), SV("3月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{4}), SV("4月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{5}), SV("5月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{6}), SV("6月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{7}), SV("7月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{8}), SV("8月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{9}), SV("9月"));
#  else  // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{1}), SV(" 1月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{2}), SV(" 2月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{3}), SV(" 3月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{4}), SV(" 4月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{5}), SV(" 5月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{6}), SV(" 6月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{7}), SV(" 7月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{8}), SV(" 8月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{9}), SV(" 9月"));
#  endif // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{10}), SV("10月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{11}), SV("11月"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{12}), SV("12月"));
#endif   // defined(__APPLE__)|| defined(_WIN32)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{13}), SV("13 is not a valid month"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month{255}), SV("255 is not a valid month"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
