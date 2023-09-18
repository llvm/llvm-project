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
#include "assert_macros.h"
#include "concat_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

#define TEST_EQUAL(OUT, EXPECTED)                                                                                      \
  TEST_REQUIRE(OUT == EXPECTED,                                                                                        \
               TEST_WRITE_CONCATENATED(                                                                                \
                   "\nExpression      ", #OUT, "\nExpected output ", EXPECTED, "\nActual output   ", OUT, '\n'));

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
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{0}}}),
             SV("0000/0 is not a valid month/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/Jan/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/Feb/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/Mar/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/Apr/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/May/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/Jun/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/Jul/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/Aug/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/Sep/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}),
             SV("0000/Oct/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}),
             SV("0000/Nov/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}),
             SV("0000/Dec/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{13}}}),
             SV("0000/13 is not a valid month/last"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{255}}}),
             SV("-32768 is not a valid year/255 is not a valid month/last"));

  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{0}}}),
             SV("0000/0 is not a valid month/last"));
#if defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/jan/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/fév/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/mar/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/avr/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/mai/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/jui/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/jul/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/aoû/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/sep/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}),
             SV("0000/oct/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}),
             SV("0000/nov/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}),
             SV("0000/déc/last"));
#else //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/janv./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/févr./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/mars/last"));
#  if defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/avr./last"));
#  else  // defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/avril/last"));
#  endif // defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/mai/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/juin/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/juil./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/août/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/sept./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}),
             SV("0000/oct./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}),
             SV("0000/nov./last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}),
             SV("0000/déc./last"));
#endif   //  defined(__APPLE__)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{13}}}),
             SV("0000/13 is not a valid month/last"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{255}}}),
             SV("-32768 is not a valid year/255 is not a valid month/last"));

  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{0}}}),
             SV("0000/0 is not a valid month/last"));
#if defined(__APPLE__) || defined(_WIN32)
#  if defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/ 1/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/ 2/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/ 3/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/ 4/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/ 5/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/ 6/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/ 7/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/ 8/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/ 9/last"));
#  else  // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/1/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/2/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/3/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/4/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/5/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/6/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/7/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/8/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/9/last"));
#  endif // defined(__APPLE__)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}),
             SV("0000/10/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}),
             SV("0000/11/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}),
             SV("0000/12/last"));
#else // defined(__APPLE__) || defined(_WIN32)
#  if defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/1月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/2月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/3月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/4月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/5月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/6月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/7月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/8月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/9月/last"));
#  else  // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{1}}}),
             SV("-32768 is not a valid year/ 1月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'767}, std::chrono::month_day_last{std::chrono::month{2}}}),
             SV("-32767/ 2月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{3}}}),
             SV("0000/ 3月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{4}}}),
             SV("1970/ 4月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{32'767}, std::chrono::month_day_last{std::chrono::month{5}}}),
             SV("32767/ 5月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{6}}}),
             SV("0000/ 6月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{7}}}),
             SV("0000/ 7月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{8}}}),
             SV("0000/ 8月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{9}}}),
             SV("0000/ 9月/last"));
#  endif // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{10}}}),
             SV("0000/10月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{11}}}),
             SV("0000/11月/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{12}}}),
             SV("0000/12月/last"));
#endif   // defined(__APPLE__) || defined(_WIN32)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{0}, std::chrono::month_day_last{std::chrono::month{13}}}),
             SV("0000/13 is not a valid month/last"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::year_month_day_last{
                 std::chrono::year{-32'768}, std::chrono::month_day_last{std::chrono::month{255}}}),
             SV("-32768 is not a valid year/255 is not a valid month/last"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
