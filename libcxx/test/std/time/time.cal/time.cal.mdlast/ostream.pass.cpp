//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-FREEBSD-FIXME

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// TODO FMT It seems GCC uses too much memory in the CI and fails.
// UNSUPPORTED: gcc-12

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// class month_day_last;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day_last& mdl);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::month_day_last mdl) {
  std::basic_stringstream<CharT> sstr;
  sstr << mdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month_day_last mdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << mdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month_day_last mdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << mdl;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{0}}) ==
         SV("0 is not a valid month/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}) == SV("Jan/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}) == SV("Feb/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}) == SV("Mar/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV("Apr/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}) == SV("May/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}) == SV("Jun/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}) == SV("Jul/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}) == SV("Aug/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}) == SV("Sep/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}) == SV("Oct/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}) == SV("Nov/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}) == SV("Dec/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{13}}) ==
         SV("13 is not a valid month/last"));
  assert(stream_c_locale<CharT>(std::chrono::month_day_last{std::chrono::month{255}}) ==
         SV("255 is not a valid month/last"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{0}}) ==
         SV("0 is not a valid month/last"));
#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}) == SV("jan/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}) == SV("fév/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}) == SV("mar/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV("avr/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}) == SV("mai/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}) == SV("jui/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}) == SV("jul/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}) == SV("aoû/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}) == SV("sep/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}) == SV("oct/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}) == SV("nov/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}) == SV("déc/last"));
#else //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}) == SV("janv./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}) == SV("févr./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}) == SV("mars/last"));
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV("avr./last"));
#  else
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV("avril/last"));
#  endif
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}) == SV("mai/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}) == SV("juin/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}) == SV("juil./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}) == SV("août/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}) == SV("sept./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}) == SV("oct./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}) == SV("nov./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}) == SV("déc./last"));
#endif //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{13}}) ==
         SV("13 is not a valid month/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_day_last{std::chrono::month{255}}) ==
         SV("255 is not a valid month/last"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{0}}) ==
         SV("0 is not a valid month/last"));
#if defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}) == SV(" 1/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}) == SV(" 2/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}) == SV(" 3/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV(" 4/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}) == SV(" 5/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}) == SV(" 6/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}) == SV(" 7/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}) == SV(" 8/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}) == SV(" 9/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}) == SV("10/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}) == SV("11/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}) == SV("12/last"));
#else    // defined(__APPLE__)
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}) == SV("1月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}) == SV("2月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}) == SV("3月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV("4月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}) == SV("5月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}) == SV("6月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}) == SV("7月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}) == SV("8月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}) == SV("9月/last"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{1}}) == SV(" 1月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{2}}) == SV(" 2月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{3}}) == SV(" 3月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{4}}) == SV(" 4月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{5}}) == SV(" 5月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{6}}) == SV(" 6月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{7}}) == SV(" 7月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{8}}) == SV(" 8月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{9}}) == SV(" 9月/last"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{10}}) == SV("10月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{11}}) == SV("11月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{12}}) == SV("12月/last"));
#endif   // defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{13}}) ==
         SV("13 is not a valid month/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_day_last{std::chrono::month{255}}) ==
         SV("255 is not a valid month/last"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
