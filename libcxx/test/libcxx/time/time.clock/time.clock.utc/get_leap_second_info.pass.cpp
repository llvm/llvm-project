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
//
// class utc_clock;

// template<class Duration>
// std::chrono::leap_second_info get_leap_second_info(const utc_time<Duration>& ut);

#include <chrono>
#include <cassert>
#include <fstream>
#include <string>
#include <string_view>

#include "test_macros.h"
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

static void write(std::string_view input) {
  static int version = 0;

  {
    std::ofstream f{tzdata};
    f << "# version " << version++ << '\n';
    std::ofstream{leap_seconds}.write(input.data(), input.size());
  }
  std::chrono::reload_tzdb();
}

template <class Duration>
static void test_leap_second_info(
    std::chrono::time_point<std::chrono::utc_clock, Duration> time, bool is_leap_second, std::chrono::seconds elapsed) {
  std::chrono::leap_second_info result = std::chrono::get_leap_second_info(time);
  TEST_REQUIRE(
      result.is_leap_second == is_leap_second && result.elapsed == elapsed,
      TEST_WRITE_CONCATENATED(
          "\nExpected output [",
          is_leap_second,
          ", ",
          elapsed,
          "]\nActual output   [",
          result.is_leap_second,
          ", ",
          result.elapsed,
          "]\n"));
}

static void test_no_leap_seconds_entries() {
  using namespace std::literals::chrono_literals;

  write("");

  test_leap_second_info(
      std::chrono::utc_seconds{std::chrono::sys_days{std::chrono::January / 1 / 1900}.time_since_epoch()}, false, 0s);
  test_leap_second_info(
      std::chrono::utc_seconds{std::chrono::sys_days{std::chrono::January / 1 / 2000}.time_since_epoch()}, false, 0s);
  test_leap_second_info(
      std::chrono::utc_seconds{std::chrono::sys_days{std::chrono::January / 1 / 3000}.time_since_epoch()}, false, 0s);
}

// Note at the time of writing all leap seconds are positive. This test uses
// fake data to test the behaviour of negative leap seconds.
static void test_negative_leap_seconds() {
  using namespace std::literals::chrono_literals;

  // Use small values for simplicity. The dates are seconds since 1.1.1900.
  write(
      R"(
1 10
60 11
120 12
180 11
240 12
300 13
360 12
)");

  // Transitions from the start of UTC.
  auto test_transition = [](std::chrono::utc_seconds time, std::chrono::seconds elapsed, bool positive) {
    if (positive) {
      // Every transition has the following tests
      // - 1ns before the start of the transition is_leap_second -> false, elapsed -> elapsed
      // -         at the start of the transition is_leap_second -> true,  elapsed -> elapsed + 1
      // - 1ns after  the start of the transition is_leap_second -> true,  elapsed -> elapsed + 1
      // - 1ns before the end   of the transition is_leap_second -> true,  elapsed -> elapsed + 1
      // -         at the end   of the transition is_leap_second -> false, elapsed -> elapsed + 1

      test_leap_second_info(time - 1ns, false, elapsed);
      test_leap_second_info(time, true, elapsed + 1s);
      test_leap_second_info(time + 1ns, true, elapsed + 1s);
      test_leap_second_info(time + 1s - 1ns, true, elapsed + 1s);
      test_leap_second_info(time + 1s, false, elapsed + 1s);
    } else {
      // Every transition has the following tests
      // - 1ns before the transition is_leap_second -> false, elapsed -> elapsed
      // -         at the transition is_leap_second -> false  elapsed -> elapsed - 1
      // - 1ns after  the transition is_leap_second -> false, elapsed -> elapsed - 1
      test_leap_second_info(time - 1ns, false, elapsed);
      test_leap_second_info(time, false, elapsed - 1s);
      test_leap_second_info(time + 1ns, false, elapsed - 1s);
    }
  };

  std::chrono::utc_seconds epoch{std::chrono::sys_days{std::chrono::January / 1 / 1900}.time_since_epoch()};
  test_leap_second_info(epoch, false, 0s);

  // The UTC times are:
  //   epoch + transition time in the database + leap seconds before the transition.
  test_transition(epoch + 60s + 0s, 0s, true);
  test_transition(epoch + 120s + 1s, 1s, true);
  test_transition(epoch + 180s + 2s, 2s, false);
  test_transition(epoch + 240s + 1s, 1s, true);
  test_transition(epoch + 300s + 2s, 2s, true);
  test_transition(epoch + 360s + 3s, 3s, false);
}

int main(int, const char**) {
  test_no_leap_seconds_entries();
  test_negative_leap_seconds();

  return 0;
}
