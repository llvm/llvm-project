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

// TODO FMT Evaluate gcc-12 status
// UNSUPPORTED: gcc-12

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// class hh_mm_ss;

// template<class charT, class traits, class Duration>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const hh_mm_ss<Duration>& hms);

#include <cassert>
#include <chrono>
#include <ratio>
#include <sstream>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT, class Duration>
static std::basic_string<CharT> stream_c_locale(std::chrono::hh_mm_ss<Duration> hms) {
  std::basic_stringstream<CharT> sstr;
  sstr << hms;
  return sstr.str();
}

template <class CharT, class Duration>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::hh_mm_ss<Duration> hms) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << hms;
  return sstr.str();
}

template <class CharT, class Duration>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::hh_mm_ss<Duration> hms) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << hms;
  return sstr.str();
}

template <class CharT>
static void test() {
  // Note std::atto can't be tested since the ratio conversion from std::atto
  // std::chrono::seconds to std::chrono::hours overflows when intmax_t is a
  // 64-bit type. This is a limitiation in the constructor of
  // std::chrono::hh_mm_ss.

  // C locale - integral power of 10 ratios
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::femto>{1'234'567'890}}) ==
         SV("00:00:00.000001234567890"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::pico>{1'234'567'890}}) ==
         SV("00:00:00.001234567890"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::nano>{1'234'567'890}}) ==
         SV("00:00:01.234567890"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::micro>{1'234'567}}) ==
         SV("00:00:01.234567"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::milli>{123'456}}) ==
         SV("00:02:03.456"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::centi>{12'345}}) ==
         SV("00:02:03.45"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::deci>{1'234}}) ==
         SV("00:02:03.4"));

  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t>{123}}) == SV("00:02:03"));

  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::deca>{-366}}) ==
         SV("-01:01:00"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::hecto>{-72}}) ==
         SV("-02:00:00"));
  assert(stream_c_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::kilo>{-86}}) ==
         SV("-23:53:20"));

  // Starting at mega it will pass one day

  // fr_FR locale - integral power of not 10 ratios
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{
             std::chrono::duration<intmax_t, std::ratio<1, 5'000'000>>{5'000}}) == SV("00:00:00,0010000"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 8'000>>{3}}) ==
         SV("00:00:00,000375"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 4'000>>{1}}) ==
         SV("00:00:00,00025"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 5'000>>{5}}) ==
         SV("00:00:00,0010"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 8>>{-4}}) ==
         SV("-00:00:00,500"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 4>>{-8}}) ==
         SV("-00:00:02,00"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 5>>{-5}}) ==
         SV("-00:00:01,0"));

  // TODO FMT Note there's no wording on the rounding
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 9>>{5}}) ==
         SV("00:00:00,555555"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 7>>{7}}) ==
         SV("00:00:01,000000"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 6>>{1}}) ==
         SV("00:00:00,166666"));
  assert(stream_fr_FR_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<intmax_t, std::ratio<1, 3>>{2}}) ==
         SV("00:00:00,666666"));

  // ja_JP locale - floating points
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{
             std::chrono::duration<long double, std::femto>{1'234'567'890.123}}) == SV("00:00:00.000001234567890"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{
             std::chrono::duration<long double, std::pico>{1'234'567'890.123}}) == SV("00:00:00.001234567890"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{
             std::chrono::duration<long double, std::nano>{1'234'567'890.123}}) == SV("00:00:01.234567890"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<double, std::micro>{1'234'567.123}}) ==
         SV("00:00:01.234567"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<double, std::milli>{123'456.123}}) ==
         SV("00:02:03.456"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<double, std::centi>{12'345.123}}) ==
         SV("00:02:03.45"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<float, std::deci>{1'234.123}}) ==
         SV("00:02:03.4"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<float>{123.123}}) == SV("00:02:03"));

  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<double, std::deca>{-366.5}}) ==
         SV("-01:01:05"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<double, std::hecto>{-72.64}}) ==
         SV("-02:01:04"));
  assert(stream_ja_JP_locale<CharT>(std::chrono::hh_mm_ss{std::chrono::duration<double, std::kilo>{-86}}) ==
         SV("-23:53:20"));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
