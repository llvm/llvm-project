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
const std::filesystem::path leap_seconds         = env.create_file("zoneinfo/leapseconds");

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
  test_exception("0", "corrupt tzdb: expected character 'l' from string 'leap', got '0' instead");
  test_exception("Leap  x", "corrupt tzdb: expected a digit");
  test_exception("Leap  1970  J", "corrupt tzdb month: invalid name");
  test_exception("Leap  1970  Jan   1   23:59:60    x", "corrupt tzdb: invalid leap second sign x");
}

static void test_leap_seconds() {
  using namespace std::chrono;

  // Test whether loading also sorts the entries in the proper order.
  const tzdb& result = parse(R"(
Leap  1973  Jan   1   23:59:60    +   S
Leap  1972  Jul   1   23:59:60    +   S
Leap  1972  Jan   1   23:59:60    +   S
Leap  1900  Jan   2   23:59:60    +   S # 2 Jan 1900 Dummy entry to test before 1970
Leap  1900  Jan   2   00:00:01    +   S # 2 Jan 1900 Dummy entry to test before 1970

Leap  1973  Jan   2   23:59:60    -   S # Fictional negative leap second
Leap  32767 Jan   1   23:59:60    +   S # Largest year accepted by the parser
)");

  assert(result.leap_seconds.size() == 6);

  assert(result.leap_seconds[0].date() == sys_seconds{sys_days{1900y / January / 2}});
  assert(result.leap_seconds[0].value() == 1s);

  assert(result.leap_seconds[1].date() == sys_seconds{sys_days{1972y / January / 1}});
  assert(result.leap_seconds[1].value() == 1s);

  assert(result.leap_seconds[2].date() == sys_seconds{sys_days{1972y / July / 1}});
  assert(result.leap_seconds[2].value() == 1s);

  assert(result.leap_seconds[3].date() == sys_seconds{sys_days{1973y / January / 1}});
  assert(result.leap_seconds[3].value() == 1s);

  assert(result.leap_seconds[4].date() == sys_seconds{sys_days{1973y / January / 2}});
  assert(result.leap_seconds[4].value() == -1s);

  assert(result.leap_seconds[5].date() ==
         sys_seconds{5764607523034234879s
                     // The database uses 1900-01-01 as epoch.
                     - std::chrono::duration_cast<std::chrono::seconds>(
                           sys_days{1970y / January / 1} - sys_days{1900y / January / 1})});
  assert(result.leap_seconds[5].value() == 1s);
}

int main(int, const char**) {
  test_invalid();
  test_leap_seconds();

  return 0;
}
