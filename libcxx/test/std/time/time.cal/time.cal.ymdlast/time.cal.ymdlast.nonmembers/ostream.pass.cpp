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

// class year_month_day_last;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year_month_day_last& ymdl);

#include <chrono>
#include <cassert>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static std::basic_string<CharT> stream_c_locale(std::chrono::year_month_day_last ymdl) {
  std::basic_stringstream<CharT> sstr;
  sstr << ymdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::year_month_day_last ymdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << ymdl;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::year_month_day_last ymdl) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << ymdl;
  return sstr.str();
}

template <class CharT>
static void test() {
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{0}}}) ==
         SV("0000/0 is not a valid month/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}) ==
         SV("-32768 is not a valid year/Jan/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}) == SV("-32767/Feb/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}) == SV("0000/Mar/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/Apr/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}) == SV("32767/May/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}) == SV("0000/Jun/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}) == SV("0000/Jul/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}) == SV("0000/Aug/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}) == SV("0000/Sep/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}) == SV("0000/Oct/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}) == SV("0000/Nov/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}) == SV("0000/Dec/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{13}}}) ==
         SV("0000/13 is not a valid month/last"));
  assert(stream_c_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{255}}}) ==
         SV("-32768 is not a valid year/255 is not a valid month/last"));

  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{0}}}) ==
         SV("0000/0 is not a valid month/last"));
#if defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}) ==
         SV("-32768 is not a valid year/jan/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}) == SV("-32767/fév/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}) == SV("0000/mar/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/avr/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}) == SV("32767/mai/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}) == SV("0000/jui/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}) == SV("0000/jul/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}) == SV("0000/aoû/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}) == SV("0000/sep/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}) == SV("0000/oct/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}) == SV("0000/nov/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}) == SV("0000/déc/last"));
#else    //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}) ==
         SV("-32768 is not a valid year/janv./last"));
  assert(
      stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
          std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}) == SV("-32767/févr./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}) == SV("0000/mars/last"));
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/avr./last"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/avril/last"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}) == SV("32767/mai/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}) == SV("0000/juin/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}) == SV("0000/juil./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}) == SV("0000/août/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}) == SV("0000/sept./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}) == SV("0000/oct./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}) == SV("0000/nov./last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}) == SV("0000/déc./last"));
#endif   //  defined(__APPLE__)
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{13}}}) ==
         SV("0000/13 is not a valid month/last"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{255}}}) ==
         SV("-32768 is not a valid year/255 is not a valid month/last"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{0}}}) ==
         SV("0000/0 is not a valid month/last"));
#if defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}) ==
         SV("-32768 is not a valid year/ 1/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}) == SV("-32767/ 2/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}) == SV("0000/ 3/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/ 4/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}) == SV("32767/ 5/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}) == SV("0000/ 6/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}) == SV("0000/ 7/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}) == SV("0000/ 8/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}) == SV("0000/ 9/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}) == SV("0000/10/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}) == SV("0000/11/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}) == SV("0000/12/last"));
#else    // defined(__APPLE__)
#  if defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}) ==
         SV("-32768 is not a valid year/1月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}) == SV("-32767/2月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}) == SV("0000/3月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/4月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}) == SV("32767/5月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}) == SV("0000/6月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}) == SV("0000/7月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}) == SV("0000/8月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}) == SV("0000/9月/last"));
#  else  // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}) ==
         SV("-32768 is not a valid year/ 1月/last"));
  assert(
      stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
          std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}) == SV("-32767/ 2月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}) == SV("0000/ 3月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}) == SV("1970/ 4月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}) == SV("32767/ 5月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}) == SV("0000/ 6月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}) == SV("0000/ 7月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}) == SV("0000/ 8月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}) == SV("0000/ 9月/last"));
#  endif // defined(_WIN32) || defined(_AIX)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}) == SV("0000/10月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}) == SV("0000/11月/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}) == SV("0000/12月/last"));
#endif   // defined(__APPLE__)
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{13}}}) ==
         SV("0000/13 is not a valid month/last"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
             std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{255}}}) ==
         SV("-32768 is not a valid year/255 is not a valid month/last"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
