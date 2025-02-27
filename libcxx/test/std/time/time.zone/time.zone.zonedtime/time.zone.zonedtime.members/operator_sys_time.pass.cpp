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
// operator sys_time<duration>() const;

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

namespace cr = std::chrono;

static void test_const_member() {
  {
    using duration   = cr::nanoseconds;
    using time_point = cr::sys_time<duration>;
    static_assert(std::is_convertible_v<time_point, cr::zoned_time<duration>>);
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = static_cast<time_point>(zt);
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = cr::nanoseconds;
    using time_point = cr::sys_time<duration>;
    static_assert(std::is_convertible_v<time_point, const cr::zoned_time<duration>>);
    const cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = static_cast<time_point>(zt);
    assert(time == time_point{duration{42}});
  }
}

static void test_duration_conversion() {
  // common_type_t<duration, seconds> -> duration
  {
    using duration   = cr::nanoseconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = static_cast<time_point>(zt);
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = cr::microseconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = static_cast<time_point>(zt);
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = cr::milliseconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = static_cast<time_point>(zt);
    assert(time == time_point{duration{42}});
  }
  // common_type_t<seconds, seconds> -> seconds
  {
    using duration   = cr::seconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<time_point> decltype(auto) time = static_cast<time_point>(zt);
    assert(time == time_point{duration{42}});
  }
  // common_type_t<duration, seconds> -> seconds
  {
    using duration   = cr::days;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<cr::sys_seconds> decltype(auto) time = static_cast<cr::sys_seconds>(zt);
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = cr::weeks;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<cr::sys_seconds> decltype(auto) time = static_cast<cr::sys_seconds>(zt);
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = cr::months;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<cr::sys_seconds> decltype(auto) time = static_cast<cr::sys_seconds>(zt);
    assert(time == time_point{duration{42}});
  }
  {
    using duration   = cr::years;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration> zt{time_point{duration{42}}};

    std::same_as<cr::sys_seconds> decltype(auto) time = static_cast<cr::sys_seconds>(zt);
    assert(time == time_point{duration{42}});
  }
}

int main(int, char**) {
  test_const_member();
  test_duration_conversion();

  return 0;
}
