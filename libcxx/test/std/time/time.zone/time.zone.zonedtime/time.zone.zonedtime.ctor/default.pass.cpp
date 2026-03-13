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
// zoned_time();

#include <chrono>
#include <concepts>
#include <type_traits>

#include "../test_offset_time_zone.h"

// Verify the results of the default constructed object,
// and whether the constructor's constraints are satisfied.
int main(int, char**) {
  {
    static_assert(std::default_initializable<std::chrono::zoned_time<std::chrono::seconds>>);
    std::chrono::zoned_time<std::chrono::seconds> zt;
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }

  static_assert(!std::default_initializable<
                std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::none>>>);

  {
    using type = offset_time_zone<offset_time_zone_flags::has_default_zone>;
    static_assert(std::default_initializable<std::chrono::zoned_time<std::chrono::seconds, type>>);

    std::chrono::zoned_time<std::chrono::seconds, type> zt;

    assert(zt.get_time_zone().offset() == std::chrono::seconds{0});
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }

  static_assert(
      !std::default_initializable<
          std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::has_locate_zone>>>);

  {
    using type = offset_time_zone<offset_time_zone_flags::both>;
    static_assert(std::default_initializable<std::chrono::zoned_time<std::chrono::seconds, type>>);

    std::chrono::zoned_time<std::chrono::seconds, type> zt;

    assert(zt.get_time_zone().offset() == std::chrono::seconds{0});
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }

  return 0;
}
