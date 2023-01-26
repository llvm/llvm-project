//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT It seems GCC uses too much memory in the CI and fails.
// UNSUPPORTED: gcc-12

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
// class weekday_indexed;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const weekday_indexed& wdi);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::weekday_indexed wdi) {
  std::basic_stringstream<CharT> sstr;
  sstr << wdi;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::weekday_indexed wdi) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << wdi;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::weekday_indexed wdi) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << wdi;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(0), 1}) == SV("Sun[1]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(1), 2}) == SV("Mon[2]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(2), 3}) == SV("Tue[3]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(3), 4}) == SV("Wed[4]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(4), 5}) == SV("Thu[5]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(5), 0}) ==
         SV("Fri[0 is not a valid index]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(6), 6}) ==
         SV("Sat[6 is not a valid index]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(7), 7}) ==
         SV("Sun[7 is not a valid index]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(8), 0}) ==
         SV("8 is not a valid weekday[0 is not a valid index]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(255), 1}) ==
         SV("255 is not a valid weekday[1]"));

#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(0), 1}) == SV("Dim[1]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(1), 2}) == SV("Lun[2]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(2), 3}) == SV("Mar[3]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(3), 4}) == SV("Mer[4]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(4), 5}) == SV("Jeu[5]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(5), 0}) ==
         SV("Ven[0 is not a valid index]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(6), 6}) ==
         SV("Sam[6 is not a valid index]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(7), 7}) ==
         SV("Dim[7 is not a valid index]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(255), 1}) ==
         SV("255 is not a valid weekday[1]"));
#else  // defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(0), 1}) == SV("dim.[1]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(1), 2}) == SV("lun.[2]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(2), 3}) == SV("mar.[3]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(3), 4}) == SV("mer.[4]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(4), 5}) == SV("jeu.[5]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(5), 0}) ==
         SV("ven.[0 is not a valid index]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(6), 6}) ==
         SV("sam.[6 is not a valid index]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(7), 7}) ==
         SV("dim.[7 is not a valid index]"));
#endif // defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(8), 0}) ==
         SV("8 is not a valid weekday[0 is not a valid index]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(255), 1}) ==
         SV("255 is not a valid weekday[1]"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(0), 1}) == SV("日[1]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(1), 2}) == SV("月[2]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(2), 3}) == SV("火[3]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(3), 4}) == SV("水[4]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(4), 5}) == SV("木[5]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(5), 0}) ==
         SV("金[0 is not a valid index]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(6), 6}) ==
         SV("土[6 is not a valid index]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(7), 7}) ==
         SV("日[7 is not a valid index]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(8), 0}) ==
         SV("8 is not a valid weekday[0 is not a valid index]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_indexed{std::chrono::weekday(255), 1}) ==
         SV("255 is not a valid weekday[1]"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
