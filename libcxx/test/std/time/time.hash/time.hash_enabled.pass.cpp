//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// Test that <chrono> provides all of the hash specializations.

#include <chrono>
#include "poisoned_hash_helper.h"
namespace chrono = std::chrono;

int main(int, char**) {
  test_hash_enabled<chrono::nanoseconds>();
  test_hash_enabled<chrono::microseconds>();
  test_hash_enabled<chrono::milliseconds>();
  test_hash_enabled<chrono::seconds>();
  test_hash_enabled<chrono::minutes>();
  test_hash_enabled<chrono::hours>();
  test_hash_enabled<chrono::days>();
  test_hash_enabled<chrono::weeks>();
  test_hash_enabled<chrono::months>();
  test_hash_enabled<chrono::years>();

  test_hash_enabled<chrono::day>();
  test_hash_enabled<chrono::month>();
  test_hash_enabled<chrono::year>();

  test_hash_enabled<chrono::weekday>();
  test_hash_enabled<chrono::weekday_indexed>();
  test_hash_enabled(chrono::weekday_last(chrono::weekday{}));

  test_hash_enabled<chrono::month_day>();
  test_hash_enabled(chrono::month_day_last(chrono::month{}));

  test_hash_enabled(chrono::month_weekday(chrono::month{}, chrono::weekday_indexed{}));
  test_hash_enabled(chrono::month_weekday_last(chrono::month{}, chrono::weekday_last(chrono::weekday{})));

  test_hash_enabled<chrono::year_month>();

  test_hash_enabled<chrono::year_month_day>();
  test_hash_enabled(chrono::year_month_day_last(chrono::year{}, chrono::month_day_last(chrono::month{})));

  test_hash_enabled<chrono::year_month_weekday>();
  test_hash_enabled(
      chrono::year_month_weekday_last(chrono::year{}, chrono::month{}, chrono::weekday_last(chrono::weekday{})));

  return 0;
}
