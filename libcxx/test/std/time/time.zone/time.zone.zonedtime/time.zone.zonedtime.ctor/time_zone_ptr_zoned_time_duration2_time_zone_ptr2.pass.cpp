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

// template<class Duration2, class TimeZonePtr2>
// zoned_time(TimeZonePtr z, const zoned_time<Duration2, TimeZonePtr2>& y);

#include <chrono>
#include <concepts>
#include <cassert>

namespace cr = std::chrono;

template <>
struct cr::zoned_traits<int> {
  static int default_zone() { return 0; }
};

int main(int, char**) {
  using ptr = const cr::time_zone*;
  ptr tz    = cr::locate_zone("UTC");

  // is_convertible_v<sys_time<Duration2>, sys_time<Duration>> is true.
  {
    using duration   = cr::microseconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration, int> zt{time_point{duration{42}}};

    using duration2   = cr::nanoseconds;
    using time_point2 = cr::sys_time<duration2>;
    static_assert(
        std::constructible_from<cr::zoned_time<duration2>, const cr::time_zone*, cr::zoned_time<duration, int>>);
    cr::zoned_time<duration2> zt2{tz, zt};

    assert(zt2.get_time_zone() == tz);
    assert(zt2.get_sys_time() == time_point2{duration2{42'000}});
  }
  {
    using duration   = cr::milliseconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration, int> zt{time_point{duration{42}}};

    using duration2   = cr::nanoseconds;
    using time_point2 = cr::sys_time<duration2>;
    static_assert(
        std::constructible_from<cr::zoned_time<duration2>, const cr::time_zone*, cr::zoned_time<duration, int>>);
    cr::zoned_time<duration2> zt2{tz, zt};

    assert(zt2.get_time_zone() == tz);
    assert(zt2.get_sys_time() == time_point2{duration2{42'000'000}});
  }
  {
    using duration   = cr::seconds;
    using time_point = cr::sys_time<duration>;
    cr::zoned_time<duration, int> zt{time_point{duration{42}}};

    using duration2   = cr::nanoseconds;
    using time_point2 = cr::sys_time<duration2>;
    static_assert(
        std::constructible_from<cr::zoned_time<duration2>, const cr::time_zone*, cr::zoned_time<duration, int>>);
    cr::zoned_time<duration2> zt2{tz, zt};

    assert(zt2.get_time_zone() == tz);
    assert(zt2.get_sys_time() == time_point2{duration2{42'000'000'000}});
  }
  // is_convertible_v<sys_time<Duration2>, sys_time<Duration>> is false.
  {
    using duration = cr::milliseconds;

    using duration2 = cr::seconds;
    static_assert(
        !std::constructible_from<cr::zoned_time<duration2>, const cr::time_zone*, cr::zoned_time<duration, int>>);
  }
  {
    using duration = cr::microseconds;

    using duration2 = cr::seconds;
    static_assert(
        !std::constructible_from<cr::zoned_time<duration2>, const cr::time_zone*, cr::zoned_time<duration, int>>);
  }
  {
    using duration = cr::nanoseconds;

    using duration2 = cr::seconds;
    static_assert(
        !std::constructible_from<cr::zoned_time<duration2>, const cr::time_zone*, cr::zoned_time<duration, int>>);
  }

  return 0;
}
