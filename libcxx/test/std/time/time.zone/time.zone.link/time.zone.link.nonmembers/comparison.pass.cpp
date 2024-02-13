//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// TODO TZDB (#81654) Enable tests
// UNSUPPORTED: c++20, c++23, c++26

// <chrono>

// bool operator==(const time_zone_link& x, const time_zone_link& y) noexcept;
// strong_ordering operator<=>(const time_zone_link& x, const time_zone_link& y) noexcept;

#include <chrono>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, const char**) {
  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  assert(tzdb.links.size() > 2);

  AssertOrderAreNoexcept<std::chrono::time_zone_link>();
  AssertOrderReturn<std::strong_ordering, std::chrono::time_zone_link>();

  assert(testOrder(tzdb.links[0], tzdb.links[1], std::strong_ordering::less));

  return 0;
}
