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

// const tzdb& get_tzdb();

#include <algorithm>
#include <cassert>
#include <chrono>

#include "test_macros.h"

int main(int, const char**) {
  const std::chrono::tzdb& db = std::chrono::get_tzdb();

  assert(!db.version.empty());

  LIBCPP_ASSERT(!db.__rules.empty());

  assert(!db.zones.empty());
  assert(std::ranges::is_sorted(db.zones));
  assert(std::ranges::adjacent_find(db.zones) == db.zones.end()); // is unique?

  assert(!db.links.empty());
  assert(std::ranges::is_sorted(db.links));
  assert(std::ranges::adjacent_find(db.links) == db.links.end()); // is unique?

  return 0;
}
