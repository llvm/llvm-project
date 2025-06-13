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

// template<class Duration, class TimeZonePtr = const time_zone*>
// class zoned_time;
//
// sys_time<duration> get_sys_time() const;

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

static void test_const_member() {
  {
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::sys_time<duration>;
    const std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
}

static void test_duration_conversion() {
  // common_type_t<duration, seconds> -> duration
  {
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::microseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::milliseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  // common_type_t<seconds, seconds> -> seconds
  {
    using duration   = std::chrono::seconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  // common_type_t<duration, seconds> -> seconds
  {
    using duration   = std::chrono::days;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<std::chrono::sys_seconds> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::weeks;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<std::chrono::sys_seconds> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::months;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<std::chrono::sys_seconds> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::years;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<std::chrono::sys_seconds> decltype(auto) time = zt.get_sys_time();
    assert(time == time_point{duration{42}});
  }
}

int main(int, char**) {
  test_const_member();
  test_duration_conversion();

  return 0;
}
