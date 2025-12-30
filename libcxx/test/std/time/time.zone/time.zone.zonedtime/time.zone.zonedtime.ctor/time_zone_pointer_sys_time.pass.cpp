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
// zoned_time(TimeZonePtr z, const sys_time<Duration>& st);

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

namespace cr = std::chrono;

int main(int, char**) {
  {
    using ptr = const cr::time_zone*;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, cr::zoned_time<cr::seconds, ptr>>);

    ptr tz = cr::locate_zone("UTC");
    cr::zoned_time<cr::seconds> zt{tz, cr::sys_seconds{cr::seconds{42}}};

    assert(zt.get_time_zone() == tz);
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{42}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::none>;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, cr::zoned_time<cr::seconds, ptr>>);

    ptr tz;
    cr::zoned_time<cr::seconds, ptr> zt{tz, cr::sys_seconds{cr::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{42}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_default_zone>;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, cr::zoned_time<cr::seconds, ptr>>);

    ptr tz;
    cr::zoned_time<cr::seconds, ptr> zt{tz, cr::sys_seconds{cr::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{42}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_locate_zone>;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, cr::zoned_time<cr::seconds, ptr>>);

    ptr tz;
    cr::zoned_time<cr::seconds, ptr> zt{tz, cr::sys_seconds{cr::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{42}});
  }
  {
    using ptr = offset_time_zone<offset_time_zone_flags::both>;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, ptr>);
    static_assert(!std::convertible_to<ptr, cr::zoned_time<cr::seconds, ptr>>);

    ptr tz;
    cr::zoned_time<cr::seconds, ptr> zt{tz, cr::sys_seconds{cr::seconds{42}}};

    assert(zt.get_time_zone().offset() == tz.offset());
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{42}});
  }

  return 0;
}
