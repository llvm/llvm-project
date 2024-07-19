//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class nonexistent_local_time
//
// template<class Duration>
// nonexistent_local_time(const local_time<Duration>& tp, const local_info& i);

#include <chrono>
#include <string_view>

#include "assert_macros.h"
#include "concat_macros.h"

template <class Duration>
static void
test(const std::chrono::local_time<Duration>& tp, const std::chrono::local_info& i, std::string_view expected) {
  std::chrono::nonexistent_local_time exception{tp, i};
  std::string_view result = exception.what();
  TEST_REQUIRE(result == expected,
               TEST_WRITE_CONCATENATED("Expected output\n", expected, "\n\nActual output\n", result, '\n'));
}

// The constructor constructs the runtime_error base class with a specific
// message. This implicitly tests what() too, since that is inherited from
// runtime_error there is no separate test for what().
int main(int, char**) {
  using namespace std::literals::chrono_literals;

  // There is no requirement on the ordering of PREV and NEXT so an "invalid"
  // gap is allowed. All tests with negative dates use the same order as
  // positive tests.

  test(std::chrono::local_time<std::chrono::nanoseconds>{-1ns},
       std::chrono::local_info{
           std::chrono::local_info::nonexistent,
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::September / 1 / 1969},
               std::chrono::sys_days{std::chrono::December / 31 / 1969} + 23h,
               0s,
               0min,
               "PREV"},
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::January / 1 / 1970},
               std::chrono::sys_days{std::chrono::March / 1 / 1970},
               1h,
               60min,
               "NEXT"}},
       R"(1969-12-31 23:59:59.999999999 is in a gap between
1969-12-31 23:00:00 PREV and
1970-01-01 01:00:00 NEXT which are both equivalent to
1969-12-31 23:00:00 UTC)");

  test(std::chrono::local_time<std::chrono::microseconds>{0us},
       std::chrono::local_info{
           std::chrono::local_info::nonexistent,
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::September / 1 / 1969},
               std::chrono::sys_days{std::chrono::December / 31 / 1969} + 23h,
               0s,
               0min,
               "PREV"},
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::January / 1 / 1970},
               std::chrono::sys_days{std::chrono::March / 1 / 1970},
               1h,
               60min,
               "NEXT"}},
       R"(1970-01-01 00:00:00.000000 is in a gap between
1969-12-31 23:00:00 PREV and
1970-01-01 01:00:00 NEXT which are both equivalent to
1969-12-31 23:00:00 UTC)");

  test(std::chrono::local_time<std::chrono::milliseconds>{1ms},
       std::chrono::local_info{
           std::chrono::local_info::nonexistent,
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::September / 1 / 1969},
               std::chrono::sys_days{std::chrono::December / 31 / 1969} + 23h,
               0s,
               0min,
               "PREV"},
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::January / 1 / 1970},
               std::chrono::sys_days{std::chrono::March / 1 / 1970},
               1h,
               60min,
               "NEXT"}},
       R"(1970-01-01 00:00:00.001 is in a gap between
1969-12-31 23:00:00 PREV and
1970-01-01 01:00:00 NEXT which are both equivalent to
1969-12-31 23:00:00 UTC)");

  test(std::chrono::local_seconds{(std::chrono::sys_days{std::chrono::January / 1 / -21970}).time_since_epoch()},
       std::chrono::local_info{
           std::chrono::local_info::nonexistent,
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::September / 1 / -21969},
               std::chrono::sys_days{std::chrono::December / 31 / -21969} + 23h,
               0s,
               0min,
               "PREV"},
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::January / 1 / -21970},
               std::chrono::sys_days{std::chrono::March / 1 / -21970},
               1h,
               60min,
               "NEXT"}},
       R"(-21970-01-01 00:00:00 is in a gap between
-21969-12-31 23:00:00 PREV and
-21970-01-01 01:00:00 NEXT which are both equivalent to
-21969-12-31 23:00:00 UTC)");

  test(
      std::chrono::local_time<std::chrono::days>{
          (std::chrono::sys_days{std::chrono::January / 1 / 21970}).time_since_epoch()},
      std::chrono::local_info{
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info{
              std::chrono::sys_days{std::chrono::September / 1 / 21969},
              std::chrono::sys_days{std::chrono::December / 31 / 21969} + 23h,
              0s,
              0min,
              "PREV"},
          std::chrono::sys_info{
              std::chrono::sys_days{std::chrono::January / 1 / 21970},
              std::chrono::sys_days{std::chrono::March / 1 / 21970},
              1h,
              60min,
              "NEXT"}},
      R"(21970-01-01 is in a gap between
21969-12-31 23:00:00 PREV and
21970-01-01 01:00:00 NEXT which are both equivalent to
21969-12-31 23:00:00 UTC)");

  test(std::chrono::local_time<std::chrono::weeks>{},
       std::chrono::local_info{
           std::chrono::local_info::nonexistent,
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::September / 1 / 1969},
               std::chrono::sys_days{std::chrono::December / 31 / 1969} + 23h,
               0s,
               0min,
               "PREV"},
           std::chrono::sys_info{
               std::chrono::sys_days{std::chrono::January / 1 / 1970},
               std::chrono::sys_days{std::chrono::March / 1 / 1970},
               1h,
               60min,
               "NEXT"}},
       R"(1970-01-01 is in a gap between
1969-12-31 23:00:00 PREV and
1970-01-01 01:00:00 NEXT which are both equivalent to
1969-12-31 23:00:00 UTC)");

  // Note months and years can not be streamed.

  return 0;
}
