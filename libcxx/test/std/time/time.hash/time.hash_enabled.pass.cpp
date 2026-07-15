//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <chrono>

// Test that <chrono> provides all of the hash specializations.

#include <chrono>
#include <type_traits>
#include "poisoned_hash_helper.h"
#include "test_macros.h"

static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::day>, std::chrono::day>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::month>, std::chrono::month>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::year>, std::chrono::year>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::weekday>, std::chrono::weekday>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::weekday_indexed>, std::chrono::weekday_indexed>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::weekday_last>, std::chrono::weekday_last>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::month_day>, std::chrono::month_day>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::month_day_last>, std::chrono::month_day_last>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::month_weekday>, std::chrono::month_weekday>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::month_weekday_last>, std::chrono::month_weekday_last>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::year_month>, std::chrono::year_month>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::year_month_day>, std::chrono::year_month_day>);
static_assert(
    std::is_nothrow_invocable_v<std::hash<std::chrono::year_month_day_last>, std::chrono::year_month_day_last>);
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::year_month_weekday>, std::chrono::year_month_weekday>);
static_assert(
    std::is_nothrow_invocable_v<std::hash<std::chrono::year_month_weekday_last>, std::chrono::year_month_weekday_last>);
#ifndef TEST_HAS_NO_EXPERIMENTAL_TZDB
static_assert(std::is_nothrow_invocable_v<std::hash<std::chrono::leap_second>, std::chrono::leap_second>);
#endif // TEST_HAS_NO_EXPERIMENTAL_TZDB

int main(int, char**) {
  test_hash_enabled<std::chrono::nanoseconds>();
  test_hash_enabled<std::chrono::microseconds>();
  test_hash_enabled<std::chrono::milliseconds>();
  test_hash_enabled<std::chrono::seconds>();
  test_hash_enabled<std::chrono::minutes>();
  test_hash_enabled<std::chrono::hours>();
  test_hash_enabled<std::chrono::days>();
  test_hash_enabled<std::chrono::weeks>();
  test_hash_enabled<std::chrono::months>();
  test_hash_enabled<std::chrono::years>();

  test_hash_enabled<std::chrono::day>();
  test_hash_enabled<std::chrono::month>();
  test_hash_enabled<std::chrono::year>();

  test_hash_enabled<std::chrono::weekday>();
  test_hash_enabled<std::chrono::weekday_indexed>();
  test_hash_enabled(std::chrono::weekday_last(std::chrono::weekday{}));

  test_hash_enabled<std::chrono::month_day>();
  test_hash_enabled(std::chrono::month_day_last(std::chrono::month{}));

  test_hash_enabled(std::chrono::month_weekday(std::chrono::month{}, std::chrono::weekday_indexed{}));

  test_hash_enabled(
      std::chrono::month_weekday_last(std::chrono::month{}, std::chrono::weekday_last(std::chrono::weekday{})));

  test_hash_enabled<std::chrono::year_month>();

  test_hash_enabled<std::chrono::year_month_day>();

  test_hash_enabled(
      std::chrono::year_month_day_last(std::chrono::year{}, std::chrono::month_day_last(std::chrono::month{})));

  test_hash_enabled<std::chrono::year_month_weekday>();

  test_hash_enabled(std::chrono::year_month_weekday_last(
      std::chrono::year{}, std::chrono::month{}, std::chrono::weekday_last(std::chrono::weekday{})));

#ifndef TEST_HAS_NO_EXPERIMENTAL_TZDB

  test_hash_enabled(std::chrono::leap_second({}, std::chrono::sys_seconds{}, std::chrono::seconds{}));

#  if !defined(TEST_HAS_NO_LOCALIZATION) && !defined(TEST_HAS_NO_TIME_ZONE_DATABASE) && !defined(TEST_HAS_NO_FILESYSTEM)

  test_hash_enabled<std::chrono::zoned_time<std::chrono::milliseconds>>();

#  endif // !defined(TEST_HAS_NO_LOCALIZATION) && !defined(TEST_HAS_NO_TIME_ZONE_DATABASE) &&
         // !defined(TEST_HAS_NO_FILESYSTEM)

#endif // TEST_HAS_NO_EXPERIMENTAL_TZDB

  return 0;
}
