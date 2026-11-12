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
// explicit zoned_time(TimeZonePtr z);

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

// Verify the results of the constructed object.
int main(int, char**) {
  {
    using ptr = const std::chrono::time_zone*;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz = std::chrono::locate_zone("UTC");
    std::chrono::zoned_time<std::chrono::seconds> zt{tz};

    assert(zt.get_time_zone() == tz);
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }

  {
    using ptr = offset_time_zone<offset_time_zone_flags::none>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz;
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_default_zone>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz;
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_locate_zone>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz;
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::both>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz;
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{});
  }

  return 0;
}
