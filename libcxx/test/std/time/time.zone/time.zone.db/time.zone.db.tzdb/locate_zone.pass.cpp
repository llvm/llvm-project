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

// struct tzdb

// const time_zone* locate_zone(string_view tz_name) const;

#include <cassert>
#include <chrono>
#include <string_view>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

static void test_zone(std::string_view zone) {
  const std::chrono::time_zone* tz = std::chrono::get_tzdb().locate_zone(zone);
  assert(tz);
  assert(tz->name() == zone);
}

static void test_link(std::string_view link, std::string_view zone) {
  const std::chrono::time_zone* tz = std::chrono::get_tzdb().locate_zone(link);
  assert(tz);
  assert(tz->name() == zone);
}

static void test_exception([[maybe_unused]] std::string_view zone) {
  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        [[maybe_unused]] std::string_view what{"tzdb: requested time zone not found"};
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD std::chrono::get_tzdb().locate_zone(zone));
}

int main(int, const char**) {
  const std::chrono::tzdb& db = std::chrono::get_tzdb();
  for (const auto& zone : db.zones)
    test_zone(zone.name());

  for (const auto& link : db.links)
    test_link(link.name(), link.target());

  test_exception("This is not a time zone");

  return 0;
}
