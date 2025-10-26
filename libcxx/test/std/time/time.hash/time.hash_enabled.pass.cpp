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
#include "poisoned_hash_helper.h"

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

  test_hash_enabled(std::chrono::leap_second({}, std::chrono::sys_seconds{}, std::chrono::seconds{}));

  return 0;
}
