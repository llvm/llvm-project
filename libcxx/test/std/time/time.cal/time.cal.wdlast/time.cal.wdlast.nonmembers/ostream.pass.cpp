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

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
// class weekday_last;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const weekday_last& wdl);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::weekday_last wdl) {
  std::basic_stringstream<CharT> sstr;
  sstr << wdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::weekday_last wdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << wdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::weekday_last wdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << wdl;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{0}}) == SV("Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{1}}) == SV("Mon[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{2}}) == SV("Tue[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{3}}) == SV("Wed[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{4}}) == SV("Thu[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{5}}) == SV("Fri[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{6}}) == SV("Sat[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{7}}) == SV("Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{8}}) ==
         SV("8 is not a valid weekday[last]"));
  assert(stream_c_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{255}}) ==
         SV("255 is not a valid weekday[last]"));

#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{0}}) == SV("Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{1}}) == SV("Lun[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{2}}) == SV("Mar[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{3}}) == SV("Mer[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{4}}) == SV("Jeu[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{5}}) == SV("Ven[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{6}}) == SV("Sam[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{7}}) == SV("Dim[last]"));
#else  // defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{0}}) == SV("dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{1}}) == SV("lun.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{2}}) == SV("mar.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{3}}) == SV("mer.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{4}}) == SV("jeu.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{5}}) == SV("ven.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{6}}) == SV("sam.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{7}}) == SV("dim.[last]"));
#endif // defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{8}}) ==
         SV("8 is not a valid weekday[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{255}}) ==
         SV("255 is not a valid weekday[last]"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{0}}) == SV("日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{1}}) == SV("月[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{2}}) == SV("火[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{3}}) == SV("水[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{4}}) == SV("木[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{5}}) == SV("金[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{6}}) == SV("土[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{7}}) == SV("日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{8}}) ==
         SV("8 is not a valid weekday[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::weekday_last{std::chrono::weekday{255}}) ==
         SV("255 is not a valid weekday[last]"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
