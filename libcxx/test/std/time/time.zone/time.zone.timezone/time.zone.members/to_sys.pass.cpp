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

// class time_zone;

// template <class _Duration>
// sys_time<common_type_t<Duration, seconds>>
//   to_sys(const local_time<Duration>& tp) const;

#include <chrono>
#include <format>
#include <cassert>
#include <string_view>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

// Tests unique conversions. To make sure the test is does not depend on changes
// in the database it uses a time zone with a fixed offset.
static void test_unique() {
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Etc/GMT+1");

  assert(tz->to_sys(std::chrono::local_time<std::chrono::nanoseconds>{-1ns}) ==
         std::chrono::sys_time<std::chrono::nanoseconds>{-1ns + 1h});

  assert(tz->to_sys(std::chrono::local_time<std::chrono::microseconds>{0us}) ==
         std::chrono::sys_time<std::chrono::microseconds>{1h});

  assert(tz->to_sys(std::chrono::local_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / -21970}).time_since_epoch()}) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / -21970}).time_since_epoch() + 1h});

  // sys_time<common_type_t<Duration, seconds>> is seconds for the larger types
  assert(tz->to_sys(std::chrono::local_time<std::chrono::days>{
             (std::chrono::sys_days{std::chrono::January / 1 / 21970}).time_since_epoch()}) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 21970}).time_since_epoch() + 1h});

  assert(tz->to_sys(std::chrono::local_time<std::chrono::weeks>{}) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() + 1h});

  // Note months and years can not be streamed; thus the function cannot be
  // instantiated for these types. (Even when there is no exception thrown.)
}

// Tests non-existant conversions.
static void test_nonexistent() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Berlin");

  // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
  // ...
  // 1 DE CE%sT 1980
  // 1 E CE%sT
  //
  // ...
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -

  // Pick an historic date where it's well known what the time zone rules were.
  // This makes it unlikely updates to the database change these rules.
  std::chrono::local_time<std::chrono::seconds> time{
      (std::chrono::sys_days{std::chrono::March / 30 / 1986} + 2h + 30min).time_since_epoch()};

  // Validates whether the database did not change.
  std::chrono::local_info info = tz->get_info(time);
  assert(info.result == std::chrono::local_info::nonexistent);

  TEST_VALIDATE_EXCEPTION(
      std::chrono::nonexistent_local_time,
      [&]([[maybe_unused]] const std::chrono::nonexistent_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-03-30 02:30:00.000000000 is in a gap between
1986-03-30 02:00:00 CET and
1986-03-30 03:00:00 CEST which are both equivalent to
1986-03-30 01:00:00 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0ns));

  TEST_VALIDATE_EXCEPTION(
      std::chrono::nonexistent_local_time,
      [&]([[maybe_unused]] const std::chrono::nonexistent_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-03-30 02:30:00.000000 is in a gap between
1986-03-30 02:00:00 CET and
1986-03-30 03:00:00 CEST which are both equivalent to
1986-03-30 01:00:00 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0us));

  TEST_VALIDATE_EXCEPTION(
      std::chrono::nonexistent_local_time,
      [&]([[maybe_unused]] const std::chrono::nonexistent_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-03-30 02:30:00.000 is in a gap between
1986-03-30 02:00:00 CET and
1986-03-30 03:00:00 CEST which are both equivalent to
1986-03-30 01:00:00 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0ms));

  TEST_VALIDATE_EXCEPTION(
      std::chrono::nonexistent_local_time,
      [&]([[maybe_unused]] const std::chrono::nonexistent_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-03-30 02:30:00 is in a gap between
1986-03-30 02:00:00 CET and
1986-03-30 03:00:00 CEST which are both equivalent to
1986-03-30 01:00:00 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0s));

#endif // TEST_HAS_NO_EXCEPTIONS
}

// Tests ambiguous conversions.
static void test_ambiguous() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Berlin");

  // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
  // ...
  // 1 DE CE%sT 1980
  // 1 E CE%sT
  //
  // ...
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -

  // Pick an historic date where it's well known what the time zone rules were.
  // This makes it unlikely updates to the database change these rules.
  std::chrono::local_time<std::chrono::seconds> time{
      (std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h + 30min).time_since_epoch()};

  // Validates whether the database did not change.
  std::chrono::local_info info = tz->get_info(time);
  assert(info.result == std::chrono::local_info::ambiguous);

  TEST_VALIDATE_EXCEPTION(
      std::chrono::ambiguous_local_time,
      [&]([[maybe_unused]] const std::chrono::ambiguous_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-09-28 02:30:00.000000000 is ambiguous.  It could be
1986-09-28 02:30:00.000000000 CEST == 1986-09-28 00:30:00.000000000 UTC or
1986-09-28 02:30:00.000000000 CET == 1986-09-28 01:30:00.000000000 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0ns));

  TEST_VALIDATE_EXCEPTION(
      std::chrono::ambiguous_local_time,
      [&]([[maybe_unused]] const std::chrono::ambiguous_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-09-28 02:30:00.000000 is ambiguous.  It could be
1986-09-28 02:30:00.000000 CEST == 1986-09-28 00:30:00.000000 UTC or
1986-09-28 02:30:00.000000 CET == 1986-09-28 01:30:00.000000 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0us));

  TEST_VALIDATE_EXCEPTION(
      std::chrono::ambiguous_local_time,
      [&]([[maybe_unused]] const std::chrono::ambiguous_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-09-28 02:30:00.000 is ambiguous.  It could be
1986-09-28 02:30:00.000 CEST == 1986-09-28 00:30:00.000 UTC or
1986-09-28 02:30:00.000 CET == 1986-09-28 01:30:00.000 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0ms));

  TEST_VALIDATE_EXCEPTION(
      std::chrono::ambiguous_local_time,
      [&]([[maybe_unused]] const std::chrono::ambiguous_local_time& e) {
        [[maybe_unused]] std::string_view what =
            R"(1986-09-28 02:30:00 is ambiguous.  It could be
1986-09-28 02:30:00 CEST == 1986-09-28 00:30:00 UTC or
1986-09-28 02:30:00 CET == 1986-09-28 01:30:00 UTC)";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("Expected exception\n", what, "\n\nActual exception\n", e.what(), '\n'));
      },
      tz->to_sys(time + 0s));

#endif // TEST_HAS_NO_EXCEPTIONS
}

// This test does the basic validations of this function. The library function
// uses `local_info get_info(const local_time<Duration>& tp)` as implementation
// detail. The get_info function does extensive testing of the data.
int main(int, char**) {
  test_unique();
  test_nonexistent();
  test_ambiguous();

  return 0;
}
