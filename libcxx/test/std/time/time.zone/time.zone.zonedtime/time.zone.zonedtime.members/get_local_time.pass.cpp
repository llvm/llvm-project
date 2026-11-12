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
// local_time<duration> get_local_time() const;

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

namespace cr = std::chrono;

static void test_callable_with_non_const_and_const_objects() {
  {
    using duration         = cr::nanoseconds;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  {
    using duration         = cr::nanoseconds;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    const cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
}

static void test_duration_conversion() {
  // common_type_t<duration, seconds> -> duration
  {
    using duration         = cr::nanoseconds;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  {
    using duration         = cr::microseconds;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  {
    using duration         = cr::milliseconds;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  // common_type_t<seconds, seconds> -> seconds
  {
    using duration         = cr::seconds;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  // common_type_t<duration, seconds> -> seconds
  {
    using duration         = cr::days;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<cr::seconds>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  {
    using duration         = cr::weeks;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<cr::seconds>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  {
    using duration         = cr::months;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<cr::seconds>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
  {
    using duration         = cr::years;
    using time_point       = cr::sys_time<duration>;
    using local_time_point = cr::local_time<cr::seconds>;
    cr::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = zt.get_local_time();
    assert(time == local_time_point{duration{42} - cr::hours{1}});
  }
}

int main(int, char**) {
  test_callable_with_non_const_and_const_objects();
  test_duration_conversion();

  return 0;
}
