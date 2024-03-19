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

// ADDITIONAL_COMPILE_FLAGS: -O0 -g

// <chrono>

// class time_zone;

// template <class _Duration>
//   sys_info get_info(const sys_time<_Duration>& time) const;

// tests the parts not validated in the public test

#include <algorithm>
#include <cassert>
#include <fstream>
#include <chrono>
#include <format>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"
#include "filesystem_test_helper.h"
#include "test_tzdb.h"

/***** ***** HELPERS ***** *****/

scoped_test_env env;
[[maybe_unused]] const std::filesystem::path dir = env.create_dir("zoneinfo");
const std::filesystem::path file                 = env.create_file("zoneinfo/tzdata.zi");

std::string_view std::chrono::__libcpp_tzdb_directory() {
  static std::string result = dir.string();
  return result;
}

static void write(std::string_view input) {
  static int version = 0;

  std::ofstream f{file};
  f << "# version " << version++ << '\n';
  f.write(input.data(), input.size());
}

static const std::chrono::tzdb& parse(std::string_view input) {
  write(input);
  return std::chrono::reload_tzdb();
}

[[nodiscard]] static std::chrono::sys_seconds to_sys_seconds(int year) {
  std::chrono::year_month_day result{std::chrono::year{year}, std::chrono::January, std::chrono::day{1}};

  return std::chrono::time_point_cast<std::chrono::seconds>(static_cast<std::chrono::sys_days>(result));
}

static void test_exception([[maybe_unused]] std::string_view input, [[maybe_unused]] std::string_view what) {
#ifndef TEST_NOEXCEPT
  const std::chrono::tzdb& tzdb    = parse(input);
  const std::chrono::time_zone* tz = tzdb.locate_zone("Format");
  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD tz->get_info(to_sys_seconds(2000)));
#endif
}

static void invalid_format() {
  test_exception(
      R"(
R F 2000 max - Jan 5 0 0 foo
Z Format 0 F %q)",
      "corrupt tzdb FORMAT field: invalid sequence '%q' found, expected %s or %z");

  test_exception(
      R"(
R F 1970 max - Jan 5 0 0 foo
Z Format 0 F %)",
      "corrupt tzdb FORMAT field: input ended with the start of the escape sequence '%'");
}

static void test_abbrev(std::string_view input, std::string_view expected) {
  const std::chrono::tzdb& tzdb    = parse(input);
  const std::chrono::time_zone* tz = tzdb.locate_zone("Format");
  std::string result               = tz->get_info(to_sys_seconds(2000)).abbrev;
  TEST_LIBCPP_REQUIRE(result == expected, TEST_WRITE_CONCATENATED("\nExpected ", expected, "\nActual ", result, '\n'));
}

// This format is valid, however is not used in the tzdata.zi.
static void percentage_z_format() {
  test_abbrev(
      R"(
R F 1999 max - Jan 5 0 0 foo
Z Format 0 F %z)",
      "+00");

  test_abbrev(
      R"(
R F 1999 max - Jan 5 0 1 foo
Z Format 0 F %z)",
      "+01");

  test_abbrev(
      R"(
R F 1999 max - Jan 5 0 -1 foo
Z Format 0 F %z)",
      "-01");

  test_abbrev(
      R"(
R F 1999 max - Jan 5 0 0 foo
Z Format 0:45 F %z)",
      "+0045");

  test_abbrev(
      R"(
R F 1999 max - Jan 5 0 -1 foo
Z Format 0:45 F %z)",
      "-0015");
}

int main(int, const char**) {
  invalid_format();
  percentage_z_format();

  return 0;
}
