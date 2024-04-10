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

// <chrono>

// class leap_second;

// constexpr sys_seconds date() const noexcept;

#include <cassert>
#include <chrono>

#include "test_macros.h"

#include "test_chrono_leap_second.h"

constexpr void test(const std::chrono::leap_second leap_second, std::chrono::sys_seconds expected) {
  std::same_as<std::chrono::sys_seconds> auto date = leap_second.date();
  assert(date == expected);
  static_assert(noexcept(leap_second.date()));
}

constexpr bool test() {
  test(test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{0}}, std::chrono::seconds{1}),
       std::chrono::sys_seconds{std::chrono::seconds{0}});

  return true;
}

int main(int, const char**) {
  test();
  static_assert(test());

  // test with the real tzdb
  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  assert(!tzdb.leap_seconds.empty());
  test(tzdb.leap_seconds[0], tzdb.leap_seconds[0].date());

  return 0;
}
