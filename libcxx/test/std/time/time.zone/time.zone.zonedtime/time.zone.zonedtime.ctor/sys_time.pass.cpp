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
// zoned_time(const sys_time<Duration>& st);

#include <chrono>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "../test_offset_time_zone.h"

static void test_construction() {
  static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds>, std::chrono::sys_seconds>);
  std::chrono::zoned_time<std::chrono::seconds> zt{std::chrono::sys_seconds{std::chrono::seconds{42}}};
  assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
  assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{42}});
}

static void test_conversion() {
  static_assert(std::convertible_to<std::chrono::sys_seconds, std::chrono::zoned_time<std::chrono::seconds>>);
  std::chrono::zoned_time<std::chrono::seconds> zt = std::chrono::sys_seconds{std::chrono::seconds{42}};
  assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
  assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{42}});
}

static void test_duration_conversion() {
  {
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::microseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::milliseconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::seconds;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == time_point{duration{42}});
  }
  {
    using duration   = std::chrono::days;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::days{42}});
  }
  {
    using duration   = std::chrono::weeks;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::weeks{42}});
  }
  {
    using duration   = std::chrono::months;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::months{42}});
  }
  {
    using duration   = std::chrono::years;
    using time_point = std::chrono::sys_time<duration>;
    std::chrono::zoned_time<duration> zt{time_point{duration{42}}};
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::years{42}});
  }
}

static void test_duration_constraints() {
  static_assert(!std::constructible_from<
                std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::none>>,
                std::chrono::sys_seconds>);

  {
    using type = offset_time_zone<offset_time_zone_flags::has_default_zone>;
    static_assert(
        std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, type>, std::chrono::sys_seconds>);

    std::chrono::zoned_time<std::chrono::seconds, type> zt = std::chrono::sys_seconds{std::chrono::seconds{42}};

    assert(zt.get_time_zone().offset() == std::chrono::seconds{0});
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{42}});
  }

  static_assert(
      !std::constructible_from<
          std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::has_locate_zone>>,
          std::chrono::sys_seconds>);

  {
    using type = offset_time_zone<offset_time_zone_flags::both>;
    static_assert(
        std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, type>, std::chrono::sys_seconds>);

    std::chrono::zoned_time<std::chrono::seconds, type> zt = std::chrono::sys_seconds{std::chrono::seconds{42}};

    assert(zt.get_time_zone().offset() == std::chrono::seconds{0});
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{42}});
  }
}

// Verify:
// - the results of the constructed object,
// - conversion construction is possible,
// - Duration is converted to zoned_time<>::duration, and
// - whether the constructor's constraints are satisfied.
int main(int, char**) {
  test_construction();
  test_conversion();
  test_duration_conversion();
  test_duration_constraints();

  return 0;
}
