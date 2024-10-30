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

// template<class charT> struct formatter<chrono::year_month_weekday_last, charT>;

#include <chrono>
#include <format>

#include <cassert>
#include <concepts>
#include <locale>
#include <iostream>
#include <type_traits>

#include "formatter_tests.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "string_literal.h"
#include "test_macros.h"

template <class CharT>
static void test_no_chrono_specs() {
  // Valid
  check(SV("1970/Jan/Mon[last]"),
        SV("{}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("*1970/Jan/Mon[last]*"),
        SV("{:*^20}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("*1970/Jan/Mon[last]"),
        SV("{:*>19}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // Invalid
  check(SV("1970/Jan/8 is not a valid weekday[last]"),
        SV("{}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check(SV("1970/0 is not a valid month/Mon[last]"),
        SV("{}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("-32768 is not a valid year/Jan/Mon[last]"),
        SV("{}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
}

template <class CharT>
static void test_invalid_values() {
  // *** Invalid weekday ***

  // Weekday name conversion
  check_exception(
      "Formatting a weekday name needs a valid weekday",
      SV("{:%a}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check_exception(
      "Formatting a weekday name needs a valid weekday",
      SV("{:%A}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});

  // Weekday conversion
  check_exception(
      "Formatting a weekday needs a valid weekday",
      SV("{:%u}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check_exception(
      "Formatting a weekday needs a valid weekday",
      SV("{:%w}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check_exception(
      "Formatting a weekday needs a valid weekday",
      SV("{:%Ou}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check_exception(
      "Formatting a weekday needs a valid weekday",
      SV("{:%Ow}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});

  // Day of year field
  check_exception(
      "Formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});

  // Month name conversion
  check(SV("Jan"),
        SV("{:%b}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check(SV("Jan"),
        SV("{:%h}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});
  check(SV("January"),
        SV("{:%B}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{8}}});

  // *** Invalid month ***

  // Weekday name conversion
  check(SV("Mon"),
        SV("{:%a}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("Monday"),
        SV("{:%A}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  // Weekday conversion
  check(SV("1"),
        SV("{:%u}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("1"),
        SV("{:%w}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("1"),
        SV("{:%Ou}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("1"),
        SV("{:%Ow}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // Day of year field
  check_exception(
      "Formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // Month name conversion
  check_exception(
      "Formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check_exception(
      "Formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check_exception(
      "Formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // *** Invalid year ***

  // Weekday name conversion
  check(SV("Mon"),
        SV("{:%a}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("Monday"),
        SV("{:%A}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // Weekday conversion
  check(SV("1"),
        SV("{:%u}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("1"),
        SV("{:%w}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("1"),
        SV("{:%Ou}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("1"),
        SV("{:%Ow}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // Day of year field
  check_exception(
      "Formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday_last{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});

  // Month name conversion
  check(SV("Jan"),
        SV("{:%b}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("Jan"),
        SV("{:%h}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
  check(SV("January"),
        SV("{:%B}"),
        std::chrono::year_month_weekday_last{
            std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_last{std::chrono::weekday{1}}});
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_invalid_values<CharT>();
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
