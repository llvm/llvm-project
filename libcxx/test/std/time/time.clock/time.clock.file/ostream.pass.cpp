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

// class system_clock;

// template<class charT, class traits, class Duration>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const sys_time<Duration>& tp);

#include <chrono>
#include <cassert>
#include <ratio>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

// Modeled after the system clock's usings, this make adapating the
// tests easier.
template <class Duration>
using file_time    = std::chrono::time_point<std::chrono::file_clock, Duration>;
using file_seconds = file_time<std::chrono::seconds>;
using file_days    = file_time<std::chrono::days>;

template <class CharT, class Duration>
static std::basic_string<CharT> stream_c_locale(file_time<Duration> time_point) {
  std::basic_stringstream<CharT> sstr;
  sstr << std::fixed << time_point;
  return sstr.str();
}

template <class CharT, class Duration>
static std::basic_string<CharT> stream_fr_FR_locale(file_time<Duration> time_point) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << std::fixed << time_point;
  return sstr.str();
}

template <class CharT, class Duration>
static std::basic_string<CharT> stream_ja_JP_locale(file_time<Duration> time_point) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << std::fixed << time_point;
  return sstr.str();
}

template <class CharT>
static void test_c() {
  using namespace std::literals::chrono_literals;

  assert(stream_c_locale<CharT>(file_time<std::chrono::nanoseconds>{946'688'523'123'456'789ns}) ==
         SV("2000-01-01 01:02:03.123456789"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::microseconds>{946'688'523'123'456us}) ==
         SV("2000-01-01 01:02:03.123456"));

  assert(stream_c_locale<CharT>(file_time<std::chrono::milliseconds>{946'684'800'123ms}) ==
         SV("2000-01-01 00:00:00.123"));
  assert(stream_c_locale<CharT>(file_seconds{1'234'567'890s}) == SV("2009-02-13 23:31:30"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::minutes>{20'576'131min}) == SV("2009-02-13 23:31:00"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::hours>{342'935h}) == SV("2009-02-13 23:00:00"));
  assert(stream_c_locale<CharT>(file_days{std::chrono::days{14'288}}) == SV("2009-02-13 00:00:00"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::weeks>{std::chrono::weeks{2041}}) == SV("2009-02-12 00:00:00"));

  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<signed char, std::ratio<2, 1>>>{
             std::chrono::duration<signed char, std::ratio<2, 1>>{60}}) == SV("1970-01-01 00:02:00"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<short, std::ratio<1, 2>>>{
             std::chrono::duration<short, std::ratio<1, 2>>{3600}}) == SV("1970-01-01 00:30:00.0"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<int, std::ratio<1, 4>>>{
             std::chrono::duration<int, std::ratio<1, 4>>{3600}}) == SV("1970-01-01 00:15:00.00"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<long, std::ratio<1, 10>>>{
             std::chrono::duration<long, std::ratio<1, 10>>{36611}}) == SV("1970-01-01 01:01:01.1"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<long long, std::ratio<1, 100>>>{
             std::chrono::duration<long long, std::ratio<1, 100>>{12'345'678'9010}}) == SV("2009-02-13 23:31:30.10"));

  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<float, std::ratio<1, 1>>>{
             std::chrono::duration<float, std::ratio<1, 1>>{123.456}}) == SV("1970-01-01 00:02:03"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<double, std::ratio<1, 10>>>{
             std::chrono::duration<double, std::ratio<1, 10>>{123.456}}) == SV("1970-01-01 00:00:12.3"));
  assert(stream_c_locale<CharT>(file_time<std::chrono::duration<long double, std::ratio<1, 100>>>{
             std::chrono::duration<long double, std::ratio<1, 100>>{123.456}}) == SV("1970-01-01 00:00:01.23"));
}

template <class CharT>
static void test_fr_FR() {
  using namespace std::literals::chrono_literals;

  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::nanoseconds>{946'688'523'123'456'789ns}) ==
         SV("2000-01-01 01:02:03,123456789"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::microseconds>{946'688'523'123'456us}) ==
         SV("2000-01-01 01:02:03,123456"));

  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::milliseconds>{946'684'800'123ms}) ==
         SV("2000-01-01 00:00:00,123"));
  assert(stream_fr_FR_locale<CharT>(file_seconds{1'234'567'890s}) == SV("2009-02-13 23:31:30"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::minutes>{20'576'131min}) == SV("2009-02-13 23:31:00"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::hours>{342'935h}) == SV("2009-02-13 23:00:00"));
  assert(stream_fr_FR_locale<CharT>(file_days{std::chrono::days{14'288}}) == SV("2009-02-13 00:00:00"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::weeks>{std::chrono::weeks{2041}}) ==
         SV("2009-02-12 00:00:00"));

  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<signed char, std::ratio<2, 1>>>{
             std::chrono::duration<signed char, std::ratio<2, 1>>{60}}) == SV("1970-01-01 00:02:00"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<short, std::ratio<1, 2>>>{
             std::chrono::duration<short, std::ratio<1, 2>>{3600}}) == SV("1970-01-01 00:30:00,0"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<int, std::ratio<1, 4>>>{
             std::chrono::duration<int, std::ratio<1, 4>>{3600}}) == SV("1970-01-01 00:15:00,00"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<long, std::ratio<1, 10>>>{
             std::chrono::duration<long, std::ratio<1, 10>>{36611}}) == SV("1970-01-01 01:01:01,1"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<long long, std::ratio<1, 100>>>{
             std::chrono::duration<long long, std::ratio<1, 100>>{12'345'678'9010}}) == SV("2009-02-13 23:31:30,10"));

  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<float, std::ratio<1, 1>>>{
             std::chrono::duration<float, std::ratio<1, 1>>{123.456}}) == SV("1970-01-01 00:02:03"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<double, std::ratio<1, 10>>>{
             std::chrono::duration<double, std::ratio<1, 10>>{123.456}}) == SV("1970-01-01 00:00:12,3"));
  assert(stream_fr_FR_locale<CharT>(file_time<std::chrono::duration<long double, std::ratio<1, 100>>>{
             std::chrono::duration<long double, std::ratio<1, 100>>{123.456}}) == SV("1970-01-01 00:00:01,23"));
}

template <class CharT>
static void test_ja_JP() {
  using namespace std::literals::chrono_literals;

  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::nanoseconds>{946'688'523'123'456'789ns}) ==
         SV("2000-01-01 01:02:03.123456789"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::microseconds>{946'688'523'123'456us}) ==
         SV("2000-01-01 01:02:03.123456"));

  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::milliseconds>{946'684'800'123ms}) ==
         SV("2000-01-01 00:00:00.123"));
  assert(stream_ja_JP_locale<CharT>(file_seconds{1'234'567'890s}) == SV("2009-02-13 23:31:30"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::minutes>{20'576'131min}) == SV("2009-02-13 23:31:00"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::hours>{342'935h}) == SV("2009-02-13 23:00:00"));
  assert(stream_ja_JP_locale<CharT>(file_days{std::chrono::days{14'288}}) == SV("2009-02-13 00:00:00"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::weeks>{std::chrono::weeks{2041}}) ==
         SV("2009-02-12 00:00:00"));

  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<signed char, std::ratio<2, 1>>>{
             std::chrono::duration<signed char, std::ratio<2, 1>>{60}}) == SV("1970-01-01 00:02:00"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<short, std::ratio<1, 2>>>{
             std::chrono::duration<short, std::ratio<1, 2>>{3600}}) == SV("1970-01-01 00:30:00.0"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<int, std::ratio<1, 4>>>{
             std::chrono::duration<int, std::ratio<1, 4>>{3600}}) == SV("1970-01-01 00:15:00.00"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<long, std::ratio<1, 10>>>{
             std::chrono::duration<long, std::ratio<1, 10>>{36611}}) == SV("1970-01-01 01:01:01.1"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<long long, std::ratio<1, 100>>>{
             std::chrono::duration<long long, std::ratio<1, 100>>{12'345'678'9010}}) == SV("2009-02-13 23:31:30.10"));

  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<float, std::ratio<1, 1>>>{
             std::chrono::duration<float, std::ratio<1, 1>>{123.456}}) == SV("1970-01-01 00:02:03"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<double, std::ratio<1, 10>>>{
             std::chrono::duration<double, std::ratio<1, 10>>{123.456}}) == SV("1970-01-01 00:00:12.3"));
  assert(stream_ja_JP_locale<CharT>(file_time<std::chrono::duration<long double, std::ratio<1, 100>>>{
             std::chrono::duration<long double, std::ratio<1, 100>>{123.456}}) == SV("1970-01-01 00:00:01.23"));
}

template <class CharT>
static void test() {
  test_c<CharT>();
  test_fr_FR<CharT>();
  test_ja_JP<CharT>();
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
