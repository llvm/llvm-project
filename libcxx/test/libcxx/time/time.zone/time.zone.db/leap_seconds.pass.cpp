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

// Tests the IANA database leap seconds parsing and operations.
// This is not part of the public tzdb interface.

#include <cassert>
#include <chrono>
#include <fstream>
#include <string>
#include <string_view>

#include "assert_macros.h"
#include "concat_macros.h"
#include "filesystem_test_helper.h"
#include "test_tzdb.h"

scoped_test_env env;
[[maybe_unused]] const std::filesystem::path dir = env.create_dir("zoneinfo");
const std::filesystem::path tzdata               = env.create_file("zoneinfo/tzdata.zi");
const std::filesystem::path leap_seconds         = env.create_file("zoneinfo/leap-seconds.list");

std::string_view std::chrono::__libcpp_tzdb_directory() {
  static std::string result = dir.string();
  return result;
}

void write(std::string_view input) {
  static int version = 0;

  std::ofstream f{tzdata};
  f << "# version " << version++ << '\n';
  std::ofstream{leap_seconds}.write(input.data(), input.size());
}

static const std::chrono::tzdb& parse(std::string_view input) {
  write(input);
  return std::chrono::reload_tzdb();
}

static void test_exception(std::string_view input, [[maybe_unused]] std::string_view what) {
  write(input);

  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD std::chrono::reload_tzdb());
}

static void test_invalid() {
  test_exception("0", "corrupt tzdb: expected a non-zero digit");

  test_exception("1", "corrupt tzdb: expected whitespace");

  test_exception("1 ", "corrupt tzdb: expected a non-zero digit");

  test_exception("5764607523034234880 2", "corrupt tzdb: integral too large");
}

static void test_leap_seconds() {
  using namespace std::chrono;

  // Test whether loading also sorts the entries in the proper order.
  const tzdb& result = parse(
      R"(
2303683200  12  # 1 Jan 1973
2287785600  11  # 1 Jul 1972
2272060800  10  # 1 Jan 1972
86400        1  # 2 Jan 1900 Dummy entry to test before 1970

# largest accepted value by the parser
5764607523034234879 2
)");

  assert(result.leap_seconds.size() == 5);

  assert(result.leap_seconds[0].date() == sys_seconds{sys_days{1900y / January / 2}});
  assert(result.leap_seconds[0].value() == 1s);

  assert(result.leap_seconds[1].date() == sys_seconds{sys_days{1972y / January / 1}});
  assert(result.leap_seconds[1].value() == 10s);

  assert(result.leap_seconds[2].date() == sys_seconds{sys_days{1972y / July / 1}});
  assert(result.leap_seconds[2].value() == 11s);

  assert(result.leap_seconds[3].date() == sys_seconds{sys_days{1973y / January / 1}});
  assert(result.leap_seconds[3].value() == 12s);

  assert(result.leap_seconds[4].date() ==
         sys_seconds{5764607523034234879s
                     // The database uses 1900-01-01 as epoch.
                     - std::chrono::duration_cast<std::chrono::seconds>(
                           sys_days{1970y / January / 1} - sys_days{1900y / January / 1})});
  assert(result.leap_seconds[4].value() == 2s);
}

int main(int, const char**) {
  test_invalid();
  test_leap_seconds();

  return 0;
}
