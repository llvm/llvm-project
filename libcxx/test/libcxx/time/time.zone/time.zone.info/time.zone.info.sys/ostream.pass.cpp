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

// <chrono>

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const sys_info& r);

// [time.zone.info.sys]
//   7 Effects: Streams out the sys_info object r in an unspecified format.
//   8 Returns: os.
//
// Tests the output produced by this function.

#include <cassert>
#include <chrono>
#include <memory>
#include <sstream>

#include "assert_macros.h"
#include "test_macros.h"
#include "make_string.h"
#include "concat_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test(std::basic_string_view<CharT> expected, std::chrono::sys_info&& info) {
  std::basic_stringstream<CharT> sstr;
  sstr << info;
  std::basic_string<CharT> output = sstr.str();

  TEST_REQUIRE(expected == output,
               TEST_WRITE_CONCATENATED("\nExpected output ", expected, "\nActual output   ", output, '\n'));
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;
  namespace tz = std::chrono;

  test(SV("[-10484-10-16 15:30:08, 14423-03-17 15:30:07) 00:00:00 0min \"TZ\""),
       tz::sys_info{tz::sys_seconds::min(), tz::sys_seconds::max(), 0s, 0min, "TZ"});

  test(SV("[1970-01-01 00:00:00, 2038-12-31 00:00:00) 12:23:45 -67min \"DMY\""),
       tz::sys_info{static_cast<tz::sys_days>(tz::year_month_day{1970y, tz::January, 1d}),
                    static_cast<tz::sys_days>(tz::year_month_day{2038y, tz::December, 31d}),
                    12h + 23min + 45s,
                    -67min,
                    "DMY"});
}

int main(int, const char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
