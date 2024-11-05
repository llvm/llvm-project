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
// zoned_time(string_view name, const sys_time<Duration>& st);

#include <chrono>
#include <cassert>
#include <concepts>
#include <string_view>

#include "../test_offset_time_zone.h"

namespace cr = std::chrono;

// Verify the results of the constructed object.
int main(int, char**) {
  {
    using ptr = const cr::time_zone*;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, std::string_view, cr::sys_seconds>);

    cr::zoned_time<cr::seconds> zt{"UTC", cr::sys_seconds{cr::seconds{99}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{99}});
  }

  {
    using ptr = offset_time_zone<offset_time_zone_flags::none>;
    static_assert(!std::constructible_from<cr::zoned_time<cr::seconds, ptr>, std::string_view, cr::sys_seconds>);
  }

  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_default_zone>;
    static_assert(!std::constructible_from<cr::zoned_time<cr::seconds, ptr>, std::string_view, cr::sys_seconds>);
  }

  {
    using ptr = offset_time_zone<offset_time_zone_flags::has_locate_zone>;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, std::string_view, cr::sys_seconds>);

    ptr tz;
    cr::zoned_time<cr::seconds, ptr> zt{"42", cr::sys_seconds{cr::seconds{99}}};

    assert(zt.get_time_zone().offset() == cr::seconds{42});
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{99}});
  }

  {
    using ptr = offset_time_zone<offset_time_zone_flags::both>;
    static_assert(std::constructible_from<cr::zoned_time<cr::seconds, ptr>, std::string_view, cr::sys_seconds>);

    ptr tz;
    cr::zoned_time<cr::seconds, ptr> zt{"42", cr::sys_seconds{cr::seconds{99}}};

    assert(zt.get_time_zone().offset() == cr::seconds{42});
    assert(zt.get_sys_time() == cr::sys_seconds{cr::seconds{99}});
  }

  return 0;
}
