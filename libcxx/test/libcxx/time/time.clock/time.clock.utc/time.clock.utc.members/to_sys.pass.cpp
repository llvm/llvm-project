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

// static sys_time<common_type_t<_Duration, seconds>>
// to_sys(const utc_time<_Duration>& __time);

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

  std::ofstream f{tzdata};
  f << "# version " << version++ << '\n';
  std::ofstream{leap_seconds}.write(input.data(), input.size());
}

template <class Duration>
static void test_leap_seconds(std::chrono::utc_time<Duration> time, std::chrono::sys_time<Duration> expected) {
  auto result = std::chrono::utc_clock::to_sys(time);
  TEST_REQUIRE(result == expected,
               TEST_WRITE_CONCATENATED("\nExpected output ", expected, "\nActual output   ", result, '\n'));
}

// Note at the time of writing all leap seconds are positive. This test uses
// fake data to test the behaviour of negative leap seconds.
int main(int, const char**) {
  using namespace std::literals::chrono_literals;

  // Use small values for simplicity. The dates are seconds since 1.1.1970.
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

  std::chrono::sys_seconds sys_epoch{std::chrono::sys_days{std::chrono::January / 1 / 1900}};
  std::chrono::utc_seconds utc_epoch{sys_epoch.time_since_epoch()};

  test_leap_seconds(utc_epoch, sys_epoch);
  auto test_transition = [](std::chrono::sys_seconds sys, std::chrono::seconds elapsed, bool positive) {
    std::chrono::utc_seconds utc = std::chrono::utc_seconds{sys.time_since_epoch()} + elapsed;
    if (positive) {
      // Every transition has the following tests
      // - 1ns before the start of the transition no adjustment needed
      // -         at the start of the transition sys is clamped at the time just prior to the moment
      //                                          of the leap second insertion. The exact value depends
      //                                          on the resolution of the result type.
      // - 1ns before the end   of the transition sys is still clamped like before
      // -         at the end   of the transition sys is 1s behind the utc time
      // - 1ns after  the end   of the transition sys is still 1s behind the utc time
      test_leap_seconds(utc - 1ns, sys - 1ns);
      test_leap_seconds(utc, sys - 1s);
      test_leap_seconds(utc + 0ns, sys - 1ns);
      test_leap_seconds(utc + 1s - 1ns, sys - 1ns);
      test_leap_seconds(utc + 1s, sys);
      test_leap_seconds(utc + 1s + 0ns, sys + 0ns);
      test_leap_seconds(utc + 1s + 1ns, sys + 1ns);
    } else {
      // Every transition has the following tests
      // - 1ns before the transition no adjustment needed
      // -         at the transition sys is 1s ahead of the utc time
      // - 1ns after  the transition sys is still 1s ahead of the utc time
      test_leap_seconds(utc - 1ns, sys - 1ns);
      test_leap_seconds(utc, sys + 1s);
      test_leap_seconds(utc + 1ns, sys + 1s + 1ns);
    }
  };

  test_transition(sys_epoch + 60s, 0s, true);
  test_transition(sys_epoch + 120s, 1s, true);
  test_transition(sys_epoch + 180s, 2s, false);
  test_transition(sys_epoch + 240s, 1s, true);
  test_transition(sys_epoch + 300s, 2s, true);
  test_transition(sys_epoch + 360s, 3s, false);

  return 0;
}
