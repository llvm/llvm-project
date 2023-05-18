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

// class month_weekday;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_weekday& mwd);

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
static std::basic_string<CharT> stream_c_locale(std::chrono::month_weekday mwd) {
  std::basic_stringstream<CharT> sstr;
  sstr << mwd;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::month_weekday mwd) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << mwd;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::month_weekday mwd) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << mwd;
  return sstr.str();
}

template <class CharT>
static void test() {
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("0 is not a valid month/Sun[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV("Jan/Mon[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV("Feb/Tue[2]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV("Mar/Wed[3]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV("Apr/Thu[4]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV("May/Fri[5]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV("Jun/Sat[6 is not a valid index]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV("Jul/Sun[7 is not a valid index]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("Aug/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("Sep/Sun[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{10}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("Oct/Sun[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{11}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("Nov/Sun[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{12}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("Dec/Sun[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("13 is not a valid month/Sun[1]"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("255 is not a valid month/8 is not a valid weekday[0 is not a valid index]"));

#if defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("0 is not a valid month/Dim[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV("jan/Lun[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV("fév/Mar[2]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV("mar/Mer[3]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV("avr/Jeu[4]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV("mai/Ven[5]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV("jui/Sam[6 is not a valid index]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV("jul/Dim[7 is not a valid index]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("aoû/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("sep/Dim[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{10}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("oct/Dim[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{11}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("nov/Dim[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{12}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("déc/Dim[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("13 is not a valid month/Dim[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("255 is not a valid month/8 is not a valid weekday[0 is not a valid index]"));
#else //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("0 is not a valid month/dim.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV("janv./lun.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV("févr./mar.[2]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV("mars/mer.[3]"));
#  if defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV("avr./jeu.[4]"));
#  else
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV("avril/jeu.[4]"));
#  endif
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV("mai/ven.[5]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV("juin/sam.[6 is not a valid index]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV("juil./dim.[7 is not a valid index]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("août/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("sept./dim.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{10}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("oct./dim.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{11}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("nov./dim.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{12}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("déc./dim.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("13 is not a valid month/dim.[1]"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("255 is not a valid month/8 is not a valid weekday[0 is not a valid index]"));
#endif //  defined(__APPLE__)

  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("0 is not a valid month/日[1]"));
#if defined(__APPLE__) || defined(_WIN32)
#  if defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV(" 1/月[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV(" 2/火[2]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV(" 3/水[3]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV(" 4/木[4]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV(" 5/金[5]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV(" 6/土[6 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV(" 7/日[7 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV(" 8/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV(" 9/日[1]"));
#  else  // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV("1/月[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV("2/火[2]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV("3/水[3]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV("4/木[4]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV("5/金[5]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV("6/土[6 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV("7/日[7 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("8/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("9/日[1]"));
#  endif // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{10}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("10/日[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{11}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("11/日[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{12}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("12/日[1]"));
#else // defined(__APPLE__) || defined(_WIN32)
#  if defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV("1月/月[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV("2月/火[2]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV("3月/水[3]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV("4月/木[4]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV("5月/金[5]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV("6月/土[6 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV("7月/日[7 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("8月/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("9月/日[1]"));
#  else  // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}}),
             SV(" 1月/月[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{2}, std::chrono::weekday_indexed{std::chrono::weekday{2}, 2}}),
             SV(" 2月/火[2]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{3}, std::chrono::weekday_indexed{std::chrono::weekday{3}, 3}}),
             SV(" 3月/水[3]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{4}, std::chrono::weekday_indexed{std::chrono::weekday{4}, 4}}),
             SV(" 4月/木[4]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{5}, std::chrono::weekday_indexed{std::chrono::weekday{5}, 5}}),
             SV(" 5月/金[5]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{6}, std::chrono::weekday_indexed{std::chrono::weekday{6}, 6}}),
             SV(" 6月/土[6 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{7}, std::chrono::weekday_indexed{std::chrono::weekday{7}, 7}}),
             SV(" 7月/日[7 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{8}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV(" 8月/8 is not a valid weekday[0 is not a valid index]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{9}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV(" 9月/日[1]"));
#  endif // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{10}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("10月/日[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{11}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("11月/日[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{12}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("12月/日[1]"));
#endif   // defined(__APPLE__) || defined(_WIN32)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}}),
             SV("13 is not a valid month/日[1]"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::month_weekday{
                 std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}}),
             SV("255 is not a valid month/8 is not a valid weekday[0 is not a valid index]"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
