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
// zoned_time(TimeZonePtr z, const local_time<Duration>& st);

#include <chrono>
#include <concepts>
#include <cassert>

#include "test_offset_time_zone.h"

int main(int, char**) {
  {
    using ptr = const std::chrono::time_zone*;
    ptr tz    = std::chrono::locate_zone("Etc/GMT+1");
    std::chrono::zoned_time<std::chrono::seconds> zt{tz, std::chrono::local_seconds{std::chrono::seconds{0}}};

    assert(zt.get_time_zone() == tz);
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::hours{1}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::none>;
    ptr tz{"60"};
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz, std::chrono::local_seconds{std::chrono::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{102}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_default_zone>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz{"60"};
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz, std::chrono::local_seconds{std::chrono::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{102}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_locate_zone>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz{"60"};
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz, std::chrono::local_seconds{std::chrono::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{102}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::both>;
    static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, std::chrono::zoned_time<std::chrono::seconds, ptr>>);

    ptr tz{"60"};
    std::chrono::zoned_time<std::chrono::seconds, ptr> zt{tz, std::chrono::local_seconds{std::chrono::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{102}});
  }

  return 0;
}
