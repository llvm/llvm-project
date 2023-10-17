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
#include "assert_macros.h"
#include "concat_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

#define TEST_EQUAL(OUT, EXPECTED)                                                                                      \
  TEST_REQUIRE(OUT == EXPECTED,                                                                                        \
               TEST_WRITE_CONCATENATED(                                                                                \
                   "\nExpression      ", #OUT, "\nExpected output ", EXPECTED, "\nActual output   ", OUT, '\n'));

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
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("0 is not a valid month/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("Jan/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV("Feb/Mon[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV("Mar/Tue[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV("Apr/Wed[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV("May/Thu[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV("Jun/Fri[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV("Jul/Sat[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV("Aug/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("Sep/8 is not a valid weekday[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("Oct/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("Nov/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("Dec/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("13 is not a valid month/Sun[last]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{255}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("255 is not a valid month/8 is not a valid weekday[last]"));

#if defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("0 is not a valid month/Dim[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("jan/Dim[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV("fév/Lun[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV("mar/Mar[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV("avr/Mer[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV("mai/Jeu[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV("jui/Ven[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV("jul/Sam[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV("aoû/Dim[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("sep/8 is not a valid weekday[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("oct/Dim[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("nov/Dim[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("déc/Dim[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("13 is not a valid month/Dim[last]"));
#else //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("0 is not a valid month/dim.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("janv./dim.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV("févr./lun.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV("mars/mar.[last]"));
#  if defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV("avr./mer.[last]"));
#  else  // defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV("avril/mer.[last]"));
#  endif // defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV("mai/jeu.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV("juin/ven.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV("juil./sam.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV("août/dim.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("sept./8 is not a valid weekday[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("oct./dim.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("nov./dim.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("déc./dim.[last]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("13 is not a valid month/dim.[last]"));
#endif   //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{255}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("255 is not a valid month/8 is not a valid weekday[last]"));

#if defined(__APPLE__) || defined(_WIN32)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("0 is not a valid month/日[last]"));
#  if defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV(" 1/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV(" 2/月[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV(" 3/火[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV(" 4/水[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV(" 5/木[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV(" 6/金[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV(" 7/土[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV(" 8/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV(" 9/8 is not a valid weekday[last]"));
#  else  // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("1/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV("2/月[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV("3/火[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV("4/水[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV("5/木[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV("6/金[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV("7/土[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV("8/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("9/8 is not a valid weekday[last]"));
#  endif // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("10/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("11/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("12/日[last]"));
#else // defined(__APPLE__) || defined(_WIN32)
#  if defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("0 is not a valid month/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("1月/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV("2月/月[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV("3月/火[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV("4月/水[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV("5月/木[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV("6月/金[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV("7月/土[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV("8月/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("9月/8 is not a valid weekday[last]"));
#  else  // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("0 is not a valid month/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV(" 1月/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{2}, std::chrono::weekday_last{std::chrono::weekday{1}}}),
             SV(" 2月/月[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{3}, std::chrono::weekday_last{std::chrono::weekday{2}}}),
             SV(" 3月/火[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{4}, std::chrono::weekday_last{std::chrono::weekday{3}}}),
             SV(" 4月/水[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{5}, std::chrono::weekday_last{std::chrono::weekday{4}}}),
             SV(" 5月/木[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{6}, std::chrono::weekday_last{std::chrono::weekday{5}}}),
             SV(" 6月/金[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{7}, std::chrono::weekday_last{std::chrono::weekday{6}}}),
             SV(" 7月/土[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{8}, std::chrono::weekday_last{std::chrono::weekday{7}}}),
             SV(" 8月/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{9}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV(" 9月/8 is not a valid weekday[last]"));
#  endif // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{10}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("10月/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{11}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("11月/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{12}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("12月/日[last]"));
#endif   // defined(__APPLE__) || defined(_WIN32)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{13}, std::chrono::weekday_last{std::chrono::weekday{0}}}),
             SV("13 is not a valid month/日[last]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday_last{
                 std::chrono::month{255}, std::chrono::weekday_last{std::chrono::weekday{8}}}),
             SV("255 is not a valid month/8 is not a valid weekday[last]"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
