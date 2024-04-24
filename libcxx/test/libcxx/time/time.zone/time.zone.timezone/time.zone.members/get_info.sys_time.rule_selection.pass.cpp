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

// class time_zone;

// template <class _Duration>
//   sys_info get_info(const sys_time<_Duration>& time) const;

// The time zone database contains of the following entries
// - Zones,
// - Rules,
// - Links, and
// - Leapseconds.
//
// The public tzdb struct stores all entries except the Rules. How
// implementations keep track of the Rules is not specified. When the sys_info
// for a time_zone is requested it needs to use the correct Rules. This lookup
// cannot rely on 'get_tzdb()` since that returns the most recently loaded
// database.
//
// A reload could change the rules of a time zone or the time zone could no
// longer be present in the current database. These two conditions are tested.
//
// It is possible the tzdb entry has been removed by the user from the tzdb_list
// after a reload. This is UB and not tested.

#include <cassert>
#include <fstream>
#include <chrono>

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

[[nodiscard]] static std::chrono::sys_seconds to_sys_seconds(
    std::chrono::year year,
    std::chrono::month month,
    std::chrono::day day,
    std::chrono::hours h   = std::chrono::hours(0),
    std::chrono::minutes m = std::chrono::minutes{0},
    std::chrono::seconds s = std::chrono::seconds{0}) {
  std::chrono::year_month_day result{year, month, day};

  return std::chrono::time_point_cast<std::chrono::seconds>(static_cast<std::chrono::sys_days>(result)) + h + m + s;
}

static void assert_equal(const std::chrono::sys_info& lhs, const std::chrono::sys_info& rhs) {
  TEST_REQUIRE(lhs.begin == rhs.begin,
               TEST_WRITE_CONCATENATED("\nBegin:\nExpected output ", lhs.begin, "\nActual output   ", rhs.begin, '\n'));
  TEST_REQUIRE(lhs.end == rhs.end,
               TEST_WRITE_CONCATENATED("\nEnd:\nExpected output ", lhs.end, "\nActual output   ", rhs.end, '\n'));
  TEST_REQUIRE(
      lhs.offset == rhs.offset,
      TEST_WRITE_CONCATENATED("\nOffset:\nExpected output ", lhs.offset, "\nActual output   ", rhs.offset, '\n'));
  TEST_REQUIRE(lhs.save == rhs.save,
               TEST_WRITE_CONCATENATED("\nSave:\nExpected output ", lhs.save, "\nActual output   ", rhs.save, '\n'));
  TEST_REQUIRE(
      lhs.abbrev == rhs.abbrev,
      TEST_WRITE_CONCATENATED("\nAbbrev:\nExpected output ", lhs.abbrev, "\nActual output   ", rhs.abbrev, '\n'));
}

/***** ***** TESTS ***** *****/

int main(int, const char**) {
  using namespace std::literals::chrono_literals;

  // DST starts on the first of March.
  const std::chrono::tzdb& tzdb_1 = parse(
      R"(
Z Test 0 -     LMT      1900
0 Rule %s

R Rule 1900 max - Mar 1 2u 1 Summer
R Rule 1900 max - Oct 1 2u 0 Winter
)");

  const std::chrono::time_zone* tz_1 = tzdb_1.locate_zone("Test");
  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1901y, std::chrono::March, 1d, 2h),
          to_sys_seconds(1901y, std::chrono::October, 1d, 2h),
          1h,
          60min,
          "Summer"),
      tz_1->get_info(to_sys_seconds(1901y, std::chrono::March, 1d, 2h)));

  // The DST start changes from the first of March to the first of April.
  const std::chrono::tzdb& tzdb_2 = parse(
      R"(
Z Test 0 -     LMT      1900
0 Rule %s

R Rule 1900 max - Apr 1 2u 1 Summer
R Rule 1900 max - Oct 1 2u 0 Winter
)");

  const std::chrono::time_zone* tz_2 = tzdb_2.locate_zone("Test");
  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1900y, std::chrono::October, 1d, 2h),
          to_sys_seconds(1901y, std::chrono::April, 1d, 2h),
          0s,
          0min,
          "Winter"),
      tz_2->get_info(to_sys_seconds(1901y, std::chrono::March, 1d, 2h)));

  // Validate when using tz_1 the DST still starts on the first of March.
  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1901y, std::chrono::March, 1d, 2h),
          to_sys_seconds(1901y, std::chrono::October, 1d, 2h),
          1h,
          60min,
          "Summer"),
      tz_1->get_info(to_sys_seconds(1901y, std::chrono::March, 1d, 2h)));

  // The zone Test is no longer present
  [[maybe_unused]] const std::chrono::tzdb& tzdb_3 = parse("Z Etc/UTC 0 - UTC");
#ifndef TEST_HAS_NO_EXCEPTIONS
  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        std::string what = "tzdb: requested time zone not found";
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD tzdb_3.locate_zone("Test"));
#endif // TEST_HAS_NO_EXCEPTIONS

  // Search the zone Test in the original version 1 of the TZDB.
  // This database should be unaffected by the removal in version 3.
  tz_1 = tzdb_1.locate_zone("Test");

  // Validate the rules still uses version 1's DST switch in March.
  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1901y, std::chrono::March, 1d, 2h),
          to_sys_seconds(1901y, std::chrono::October, 1d, 2h),
          1h,
          60min,
          "Summer"),
      tz_1->get_info(to_sys_seconds(1901y, std::chrono::March, 1d, 2h)));

  return 0;
}
