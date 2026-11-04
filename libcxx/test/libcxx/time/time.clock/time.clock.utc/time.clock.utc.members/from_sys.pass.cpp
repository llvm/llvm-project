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
// static utc_time<common_type_t<Duration, seconds>>
// from_sys(const sys_time<Duration>& time);

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
static void
test_leap_seconds(std::chrono::time_point<std::chrono::system_clock, Duration> time, std::chrono::seconds expected) {
  auto utc  = std::chrono::utc_clock::from_sys(time);
  auto diff = utc.time_since_epoch() - time.time_since_epoch();
  TEST_REQUIRE(
      diff == expected,
      TEST_WRITE_CONCATENATED("\tTime: ", time, "\nExpected output ", expected, "\nActual output   ", diff, '\n'));
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

  std::chrono::sys_days epoch = {std::chrono::January / 1 / 1900};
  test_leap_seconds(epoch, 0s);

  test_leap_seconds(epoch + 60s - 1ns, 0s);
  test_leap_seconds(epoch + 60s, 1s);
  test_leap_seconds(epoch + 60s + 1ns, 1s);

  test_leap_seconds(epoch + 120s - 1ns, 1s);
  test_leap_seconds(epoch + 120s, 2s);
  test_leap_seconds(epoch + 120s + 1ns, 2s);

  test_leap_seconds(epoch + 180s - 1ns, 2s);
  test_leap_seconds(epoch + 180s, 1s);
  test_leap_seconds(epoch + 180s + 1ns, 1s);

  test_leap_seconds(epoch + 240s - 1ns, 1s);
  test_leap_seconds(epoch + 240s, 2s);
  test_leap_seconds(epoch + 240s + 1ns, 2s);

  test_leap_seconds(epoch + 300s - 1ns, 2s);
  test_leap_seconds(epoch + 300s, 3s);
  test_leap_seconds(epoch + 300s + 1ns, 3s);

  test_leap_seconds(epoch + 360s - 1ns, 3s);
  test_leap_seconds(epoch + 360s, 2s);
  test_leap_seconds(epoch + 360s + 1ns, 2s);

  return 0;
}
