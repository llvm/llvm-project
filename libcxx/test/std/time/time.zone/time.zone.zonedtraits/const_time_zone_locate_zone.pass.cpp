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

// template<> struct zoned_traits<const time_zone*>;

// static const time_zone* locate_zone(string_view name);

#include <chrono>
#include <cassert>
#include <concepts>

#include "assert_macros.h"

static void test(std::string_view name) {
  std::same_as<const std::chrono::time_zone*> decltype(auto) tz =
      std::chrono::zoned_traits<const std::chrono::time_zone*>::locate_zone(name);

  const std::chrono::time_zone* expected = std::chrono::locate_zone(name);
  assert(tz == expected);
}

int main(int, char**) {
  test("UTC");
  test("Europe/Berlin");
  test("Asia/Hong_Kong");

  TEST_THROWS_TYPE(std::runtime_error,
                   TEST_IGNORE_NODISCARD std::chrono::zoned_traits<const std::chrono::time_zone*>::locate_zone(
                       "there_is_no_time_zone_with_this_name"));

  return 0;
}
