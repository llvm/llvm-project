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

// const tzdb& get_tzdb_list();

#include <chrono>

#include <iterator>
#include <cassert>

#include "test_macros.h"

int main(int, const char**) {
  const std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();

  assert(!list.front().version.empty());
  assert(std::distance(list.begin(), list.end()) == 1);
  assert(std::distance(list.cbegin(), list.cend()) == 1);

  return 0;
}
