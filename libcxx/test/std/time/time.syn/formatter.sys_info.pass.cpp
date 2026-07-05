//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// XFAIL: libcpp-has-no-experimental-tzdb

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
//
// template<class charT> struct formatter<chrono::sys_info, charT>;

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
// [time.zone.info.sys]/7
//   Effects: Streams out the sys_info object r in an unspecified format.
#ifdef _LIBCPP_VERSION
  using namespace std::literals::chrono_literals;
  namespace tz = std::chrono;

  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output

  check(SV("[-10484-10-16 15:30:08, 14423-03-17 15:30:07) 00:00:00 0min \"TZ\""),
        SV("{}"),
        tz::sys_info{tz::sys_seconds::min(), tz::sys_seconds::max(), 0s, 0min, "TZ"});

  check(SV("[1970-01-01 00:00:00, 2038-12-31 00:00:00) 12:23:45 -67min \"DMY\""),
        SV("{}"),
        tz::sys_info{static_cast<tz::sys_days>(tz::year_month_day{1970y, tz::January, 1d}),
                     static_cast<tz::sys_days>(tz::year_month_day{2038y, tz::December, 31d}),
                     12h + 23min + 45s,
                     -67min,
                     "DMY"});

  std::locale::global(std::locale::classic());
#endif // _LIBCPP_VERSION
}

template <class CharT>
static void test_valid_values() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%z='%z'%t%%Ez='%Ez'%t%%Oz='%Oz'%t%%Z='%Z'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%z='%z'%t%%Ez='%Ez'%t%%Oz='%Oz'%t%%Z='%Z'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%z='-0200'\t%Ez='-02:00'\t%Oz='-02:00'\t%Z='NEG'\n"),
        fmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, -2h, 0min, "NEG"});

  check(SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='ZERO'\n"),
        fmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 0s, 0min, "ZERO"});

  check(SV("%z='+1115'\t%Ez='+11:15'\t%Oz='+11:15'\t%Z='POS'\n"),
        fmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 11h + 15min, 0min, "POS"});

  // Use the global locale (fr_FR)
  check(SV("%z='-0200'\t%Ez='-02:00'\t%Oz='-02:00'\t%Z='NEG'\n"),
        lfmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, -2h, 0min, "NEG"});

  check(SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='ZERO'\n"),
        lfmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 0s, 0min, "ZERO"});

  check(SV("%z='+1115'\t%Ez='+11:15'\t%Oz='+11:15'\t%Z='POS'\n"),
        lfmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 11h + 15min, 0min, "POS"});

  // Use supplied locale (ja_JP).
  check(loc,
        SV("%z='-0200'\t%Ez='-02:00'\t%Oz='-02:00'\t%Z='NEG'\n"),
        lfmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, -2h, 0min, "NEG"});

  check(loc,
        SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='ZERO'\n"),
        lfmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 0s, 0min, "ZERO"});

  check(loc,
        SV("%z='+1115'\t%Ez='+11:15'\t%Oz='+11:15'\t%Z='POS'\n"),
        lfmt,
        std::chrono::sys_info{std::chrono::sys_seconds{0s}, std::chrono::sys_seconds{0s}, 11h + 15min, 0min, "POS"});

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();

  check_invalid_types<CharT>({SV("z"), SV("Z"), SV("Ez"), SV("Oz")}, std::chrono::sys_info{});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
