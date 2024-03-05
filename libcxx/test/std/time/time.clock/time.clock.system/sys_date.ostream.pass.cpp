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

// TODO FMT Investigate Windows issues.
// XFAIL: msvc

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// class system_Clock;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const sys_days& dp);

#include <cassert>
#include <chrono>
#include <ratio>
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
static std::basic_string<CharT> stream_c_locale(const std::chrono::sys_days& dp) {
  std::basic_stringstream<CharT> sstr;
  sstr << dp;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_fr_FR_locale(const std::chrono::sys_days& dp) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << dp;
  return sstr.str();
}

template <class CharT>
static std::basic_string<CharT> stream_ja_JP_locale(const std::chrono::sys_days& dp) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << dp;
  return sstr.str();
}

template <class CharT>
static void test() {
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{-32'768}, std::chrono::month{1}, std::chrono::day{1}}}),
             SV("-32768-01-01 is not a valid date"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{1}}}),
             SV("1970-01-01"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{2000}, std::chrono::month{2}, std::chrono::day{29}}}),
             SV("2000-02-29"));

#if defined(_AIX)
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV("+32767-12-31"));
#elif defined(_WIN32) // defined(_AIX)
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV(""));
#else                 //  defined(_AIX)
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV("32767-12-31"));
#endif                // defined(_AIX)

  // multiples of days are considered days.
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{3}}),
             SV("1970-01-22"));
  TEST_EQUAL(stream_c_locale<CharT>(std::chrono::sys_time<std::chrono::duration<int, std::ratio<30 * 86400>>>{
                 std::chrono::duration<int, std::ratio<30 * 86400>>{1}}),
             SV("1970-01-31"));

  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{-32'768}, std::chrono::month{1}, std::chrono::day{1}}}),
             SV("-32768-01-01 is not a valid date"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{1}}}),
             SV("1970-01-01"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{2000}, std::chrono::month{2}, std::chrono::day{29}}}),
             SV("2000-02-29"));
#if defined(_AIX)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV("+32767-12-31"));
#elif defined(_WIN32) // defined(_AIX)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV(""));
#else                 // defined(_AIX)
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV("32767-12-31"));
#endif                // defined(_AIX)

  // multiples of days are considered days.
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{3}}),
             SV("1970-01-22"));
  TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::sys_time<std::chrono::duration<int, std::ratio<30 * 86400>>>{
                 std::chrono::duration<int, std::ratio<30 * 86400>>{1}}),
             SV("1970-01-31"));

  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{-32'768}, std::chrono::month{1}, std::chrono::day{1}}}),
             SV("-32768-01-01 is not a valid date"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{1}}}),
             SV("1970-01-01"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{2000}, std::chrono::month{2}, std::chrono::day{29}}}),
             SV("2000-02-29"));
#if defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV("+32767-12-31"));
#elif defined(_WIN32) // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV(""));
#else                 // defined(_AIX)
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_days{
                 std::chrono::year_month_day{std::chrono::year{32'767}, std::chrono::month{12}, std::chrono::day{31}}}),
             SV("32767-12-31"));
#endif                // defined(_AIX)

  // multiples of days are considered days.
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{3}}),
             SV("1970-01-22"));
  TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::sys_time<std::chrono::duration<int, std::ratio<30 * 86400>>>{
                 std::chrono::duration<int, std::ratio<30 * 86400>>{1}}),
             SV("1970-01-31"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
