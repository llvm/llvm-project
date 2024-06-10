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
// zoned_time(TimeZonePtr z, const zoned_time<Duration2, TimeZonePtr2>& y, choose);

#include <chrono>
#include <concepts>
#include <cassert>

template <>
struct std::chrono::zoned_traits<int> {
  static int default_zone() { return 0; }
};

int main(int, char**) {
  using ptr = const std::chrono::time_zone*;
  ptr tz    = std::chrono::locate_zone("UTC");

  // is_convertible_v<sys_time<Duration2>, sys_time<Duration>> is true.
  {
    using duration   = std::chrono::microseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration, int> zt{time_point{duration{42}}};

    using duration2   = std::chrono::nanoseconds;
    using time_point2 = std::chrono::sys_time<duration2>;
    static_assert(std::constructible_from<std::chrono::zoned_time<duration2>,
                                          const std::chrono::time_zone*,
                                          std::chrono::zoned_time<duration, int>,
                                          std::chrono::choose>);
    std::chrono::zoned_time<duration2> zt2{tz, zt, std::chrono::choose::earliest};

    assert(zt2.get_time_zone() == tz);
    assert(zt2.get_sys_time() == time_point2{duration2{42'000}});
  }
  {
    using duration   = std::chrono::milliseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration, int> zt{time_point{duration{42}}};

    using duration2   = std::chrono::nanoseconds;
    using time_point2 = std::chrono::sys_time<duration2>;
    static_assert(std::constructible_from<std::chrono::zoned_time<duration2>,
                                          const std::chrono::time_zone*,
                                          std::chrono::zoned_time<duration, int>,
                                          std::chrono::choose>);
    std::chrono::zoned_time<duration2> zt2{tz, zt, std::chrono::choose::earliest};

    assert(zt2.get_time_zone() == tz);
    assert(zt2.get_sys_time() == time_point2{duration2{42'000'000}});
  }
  {
    using duration   = std::chrono::seconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration, int> zt{time_point{duration{42}}};

    using duration2   = std::chrono::nanoseconds;
    using time_point2 = std::chrono::sys_time<duration2>;
    static_assert(std::constructible_from<std::chrono::zoned_time<duration2>,
                                          const std::chrono::time_zone*,
                                          std::chrono::zoned_time<duration, int>,
                                          std::chrono::choose>);
    std::chrono::zoned_time<duration2> zt2{tz, zt, std::chrono::choose::earliest};

    assert(zt2.get_time_zone() == tz);
    assert(zt2.get_sys_time() == time_point2{duration2{42'000'000'000}});
  }
  // is_convertible_v<sys_time<Duration2>, sys_time<Duration>> is false.
  {
    using duration = std::chrono::milliseconds;

    using duration2 = std::chrono::seconds;
    static_assert(!std::constructible_from<std::chrono::zoned_time<duration2>,
                                           const std::chrono::time_zone*,
                                           std::chrono::zoned_time<duration, int>,
                                           std::chrono::choose>);
  }
  {
    using duration = std::chrono::microseconds;

    using duration2 = std::chrono::seconds;
    static_assert(!std::constructible_from<std::chrono::zoned_time<duration2>,
                                           const std::chrono::time_zone*,
                                           std::chrono::zoned_time<duration, int>,
                                           std::chrono::choose>);
  }
  {
    using duration = std::chrono::nanoseconds;

    using duration2 = std::chrono::seconds;
    static_assert(!std::constructible_from<std::chrono::zoned_time<duration2>,
                                           const std::chrono::time_zone*,
                                           std::chrono::zoned_time<duration, int>,
                                           std::chrono::choose>);
  }

  return 0;
}
