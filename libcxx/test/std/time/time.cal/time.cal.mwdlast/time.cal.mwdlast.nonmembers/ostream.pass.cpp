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

// class month_weekday_last;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_weekday_last& mwdl);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::month_weekday_last mwdl) {
  std::basic_stringstream<CharT> sstr;
  sstr << mwdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month_weekday_last mwdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << mwdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month_weekday_last mwdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << mwdl;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("0 is not a valid month/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("Jan/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}) == SV("Feb/Mon[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}) == SV("Mar/Tue[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV("Apr/Wed[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}) == SV("May/Thu[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}) == SV("Jun/Fri[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}) == SV("Jul/Sat[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}) == SV("Aug/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("Sep/8 is not a valid weekday[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("Oct/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("Nov/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("Dec/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("13 is not a valid month/Sun[last]"));
  assert(stream_c_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{255}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("255 is not a valid month/8 is not a valid weekday[last]"));

#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("0 is not a valid month/Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("jan/Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}) == SV("fév/Lun[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}) == SV("mar/Mar[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV("avr/Mer[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}) == SV("mai/Jeu[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}) == SV("jui/Ven[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}) == SV("jul/Sam[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}) == SV("aoû/Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("sep/8 is not a valid weekday[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("oct/Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("nov/Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("déc/Dim[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("13 is not a valid month/Dim[last]"));
#else    //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("0 is not a valid month/dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("janv./dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}) == SV("févr./lun.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}) == SV("mars/mar.[last]"));
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV("avr./mer.[last]"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV("avril/mer.[last]"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}) == SV("mai/jeu.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}) == SV("juin/ven.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}) == SV("juil./sam.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}) == SV("août/dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("sept./8 is not a valid weekday[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("oct./dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("nov./dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("déc./dim.[last]"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("13 is not a valid month/dim.[last]"));
#endif   //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{255}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("255 is not a valid month/8 is not a valid weekday[last]"));

#if defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("0 is not a valid month/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV(" 1/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}) == SV(" 2/月[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}) == SV(" 3/火[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV(" 4/水[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}) == SV(" 5/木[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}) == SV(" 6/金[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}) == SV(" 7/土[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}) == SV(" 8/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV(" 9/8 is not a valid weekday[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("10/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("11/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("12/日[last]"));
#else    // defined(__APPLE__)
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("0 is not a valid month/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("1月/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}) == SV("2月/月[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}) == SV("3月/火[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV("4月/水[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}) == SV("5月/木[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}) == SV("6月/金[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}) == SV("7月/土[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}) == SV("8月/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("9月/8 is not a valid weekday[last]"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("0 is not a valid month/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV(" 1月/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}) == SV(" 2月/月[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}) == SV(" 3月/火[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}) == SV(" 4月/水[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}) == SV(" 5月/木[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}) == SV(" 6月/金[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}) == SV(" 7月/土[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}) == SV(" 8月/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV(" 9月/8 is not a valid weekday[last]"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("10月/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("11月/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}) == SV("12月/日[last]"));
#endif   // defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}) ==
         SV("13 is not a valid month/日[last]"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
             std::chrono::month{255}, std::chrono::weekday_last{std::chrono::weekday{8}}}) ==
         SV("255 is not a valid month/8 is not a valid weekday[last]"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
