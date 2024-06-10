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
// explicit operator local_time<duration>() const;

#include <chrono>
#include <concepts>

#include "test_offset_time_zone.h"

static void test_const_member() {
  {
    using duration         = std::chrono::nanoseconds;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    static_assert(!std::is_convertible_v<local_time_point, std::chrono::zoned_time<duration>>);
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  {
    using duration         = std::chrono::nanoseconds;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    static_assert(!std::is_convertible_v<local_time_point, const std::chrono::zoned_time<duration>>);
    const std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
}

static void test_duration_conversion() {
  // common_type_t<duration, seconds> -> duration
  {
    using duration         = std::chrono::nanoseconds;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  {
    using duration         = std::chrono::microseconds;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  {
    using duration         = std::chrono::milliseconds;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  // common_type_t<seconds, seconds> -> seconds
  {
    using duration         = std::chrono::seconds;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  // common_type_t<duration, seconds> -> seconds
  {
    using duration         = std::chrono::days;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<std::chrono::seconds>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  {
    using duration         = std::chrono::weeks;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<std::chrono::seconds>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  {
    using duration         = std::chrono::months;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<std::chrono::seconds>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
  {
    using duration         = std::chrono::years;
    using time_point       = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<std::chrono::seconds>;
    std::chrono::zoned_time<duration> zt{"Etc/GMT+1", time_point{duration{42}}};

    std::same_as<local_time_point> decltype(auto) time = static_cast<local_time_point>(zt);
    assert(time == local_time_point{duration{42} - std::chrono::hours{1}});
  }
}

int main(int, char**) {
  test_const_member();
  test_duration_conversion();

  return 0;
}
