//===----------------------------------------------------------------------===//
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

// XFAIL: libcpp-has-no-experimental-tzdb

// REQUIRES: locale.fr_FR.UTF-8

// <chrono>
//
// template<class charT> struct formatter<chrono::local_info, charT>;

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
#include "test_macros.h"

template <class CharT>
static void test_no_chrono_specs() {
// This test libc++ specific due to
// [time.zone.info.local]/3
//   Effects: Streams out the local_info object r in an unspecified format.
#ifdef _LIBCPP_VERSION
  using namespace std::literals::chrono_literals;
  namespace tz = std::chrono;

  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output

  // result values matching the "known" results
  check(SV("unique: "
           "{[-10484-10-16 15:30:08, 14423-03-17 15:30:07) 00:00:00 0min \"TZ\", "
           "[1970-01-01 00:00:00, 1970-01-01 00:00:00) 00:00:00 0min \"\"}"),
        SV("{}"),
        tz::local_info{tz::local_info::unique,
                       tz::sys_info{tz::sys_seconds::min(), tz::sys_seconds::max(), 0s, 0min, "TZ"},
                       tz::sys_info{}});

  check(SV("non-existent: "
           "{[1970-01-01 00:00:00, 2038-12-31 00:00:00) 12:23:45 -67min \"NEG\", "
           "[1970-01-01 00:00:00, 2038-12-31 00:00:00) -12:23:45 67min \"POS\"}"),
        SV("{}"),
        tz::local_info{
            tz::local_info::nonexistent,
            tz::sys_info{static_cast<tz::sys_days>(tz::year_month_day{1970y, tz::January, 1d}),
                         static_cast<tz::sys_days>(tz::year_month_day{2038y, tz::December, 31d}),
                         12h + 23min + 45s,
                         -67min,
                         "NEG"},
            tz::sys_info{static_cast<tz::sys_days>(tz::year_month_day{1970y, tz::January, 1d}),
                         static_cast<tz::sys_days>(tz::year_month_day{2038y, tz::December, 31d}),
                         -(12h + 23min + 45s),
                         67min,
                         "POS"}});

  check(SV("ambiguous: "
           "{[1970-01-01 00:00:00, 2038-12-31 00:00:00) 12:23:45 -67min \"NEG\", "
           "[1970-01-01 00:00:00, 2038-12-31 00:00:00) -12:23:45 67min \"POS\"}"),
        SV("{}"),
        tz::local_info{
            tz::local_info::ambiguous,
            tz::sys_info{static_cast<tz::sys_days>(tz::year_month_day{1970y, tz::January, 1d}),
                         static_cast<tz::sys_days>(tz::year_month_day{2038y, tz::December, 31d}),
                         12h + 23min + 45s,
                         -67min,
                         "NEG"},
            tz::sys_info{static_cast<tz::sys_days>(tz::year_month_day{1970y, tz::January, 1d}),
                         static_cast<tz::sys_days>(tz::year_month_day{2038y, tz::December, 31d}),
                         -(12h + 23min + 45s),
                         67min,
                         "POS"}});

  // result values not matching the "known" results
  check(
      SV("unspecified result (-1): "
         "{[-10484-10-16 15:30:08, 14423-03-17 15:30:07) 00:00:00 0min \"TZ\", "
         "[1970-01-01 00:00:00, 1970-01-01 00:00:00) 00:00:00 0min \"\"}"),
      SV("{}"),
      tz::local_info{-1, tz::sys_info{tz::sys_seconds::min(), tz::sys_seconds::max(), 0s, 0min, "TZ"}, tz::sys_info{}});
  check(
      SV("unspecified result (3): "
         "{[-10484-10-16 15:30:08, 14423-03-17 15:30:07) 00:00:00 0min \"TZ\", "
         "[1970-01-01 00:00:00, 1970-01-01 00:00:00) 00:00:00 0min \"\"}"),
      SV("{}"),
      tz::local_info{3, tz::sys_info{tz::sys_seconds::min(), tz::sys_seconds::max(), 0s, 0min, "TZ"}, tz::sys_info{}});

  std::locale::global(std::locale::classic());
#endif // _LIBCPP_VERSION
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();

  check_invalid_types<CharT>({}, std::chrono::local_info{0, {}, {}});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
