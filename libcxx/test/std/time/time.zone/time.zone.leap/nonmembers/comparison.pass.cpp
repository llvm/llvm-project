//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// TODO TZDB investigate why this fails with GCC
// UNSUPPORTED: gcc-14

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class leap_second;

//constexpr bool operator==(const leap_second& x, const leap_second& y);           // C++20
//constexpr strong_ordering operator<=>(const leap_second& x, const leap_second& y);
//
//template<class Duration>
//  constexpr bool operator==(const leap_second& x, const sys_time<Duration>& y);
//template<class Duration>
//  constexpr bool operator< (const leap_second& x, const sys_time<Duration>& y);
//template<class Duration>
//  constexpr bool operator< (const sys_time<Duration>& x, const leap_second& y);
//template<class Duration>
//  constexpr bool operator> (const leap_second& x, const sys_time<Duration>& y);
//template<class Duration>
//  constexpr bool operator> (const sys_time<Duration>& x, const leap_second& y);
//template<class Duration>
//  constexpr bool operator<=(const leap_second& x, const sys_time<Duration>& y);
//template<class Duration>
//  constexpr bool operator<=(const sys_time<Duration>& x, const leap_second& y);
//template<class Duration>
//  constexpr bool operator>=(const leap_second& x, const sys_time<Duration>& y);
//template<class Duration>
//  constexpr bool operator>=(const sys_time<Duration>& x, const leap_second& y);
//template<class Duration>
//  requires three_way_comparable_with<sys_seconds, sys_time<Duration>>
//  constexpr auto operator<=>(const leap_second& x, const sys_time<Duration>& y);

#include <cassert>
#include <chrono>

#include "test_macros.h"
#include "test_comparisons.h"

#include "test_chrono_leap_second.h"

constexpr void test_comparison(const std::chrono::leap_second lhs, const std::chrono::leap_second rhs) {
  AssertOrderReturn<std::strong_ordering, std::chrono::leap_second>();
  assert(testOrder(lhs, rhs, std::strong_ordering::less));

  AssertOrderReturn<std::strong_ordering, std::chrono::leap_second, std::chrono::sys_seconds>();
  assert(testOrder(lhs, rhs.date(), std::strong_ordering::less));

  AssertOrderReturn<std::strong_ordering, std::chrono::sys_seconds, std::chrono::leap_second>();
  assert(testOrder(lhs.date(), rhs, std::strong_ordering::less));
}

constexpr bool test() {
  test_comparison(test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{0}}, std::chrono::seconds{1}),
                  test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{1}}, std::chrono::seconds{2}));

  return true;
}

int main(int, const char**) {
  test();
  static_assert(test());

  // test with the real tzdb
  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  assert(tzdb.leap_seconds.size() > 2);
  test_comparison(tzdb.leap_seconds[0], tzdb.leap_seconds[1]);

  return 0;
}
