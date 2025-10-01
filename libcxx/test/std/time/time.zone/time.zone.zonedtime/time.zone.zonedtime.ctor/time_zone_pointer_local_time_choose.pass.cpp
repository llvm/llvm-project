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
// zoned_time(TimeZonePtr z, const local_time<Duration>& st, choose c);

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

namespace cr = std::chrono;

int main(int, char**) {
  // Tests unique conversions. To make sure the test is does not depend on changes
  // in the database it uses a time zone with a fixed offset.
  {
    using ptr = const cr::time_zone*;
    ptr tz    = cr::locate_zone("Etc/GMT+1");
    cr::zoned_time<cr::seconds> zt{tz, cr::local_seconds{cr::seconds{0}}, cr::choose::earliest};

    assert(zt.get_time_zone() == tz);
    assert(zt.get_sys_time() == cr::sys_seconds{cr::hours{1}});
  }

  // Tests ambiguous conversions.
  {
    // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
    // ...
    // 1 DE CE%sT 1980
    // 1 E CE%sT
    //
    // ...
    // R E 1981 ma - Mar lastSu 1u 1 S
    // R E 1996 ma - O lastSu 1u 0 -

    using namespace std::literals::chrono_literals;
    using ptr = const cr::time_zone*;
    ptr tz    = cr::locate_zone("Europe/Berlin");
    {
      cr::zoned_time<cr::seconds> zt{
          tz,
          cr::local_seconds{(cr::sys_days{cr::September / 28 / 1986} + 2h + 30min).time_since_epoch()},
          cr::choose::earliest};

      assert(zt.get_time_zone() == tz);
      assert(zt.get_sys_time() == cr::sys_days{cr::September / 28 / 1986} + 0h + 30min);
    }
    {
      cr::zoned_time<cr::seconds> zt{
          tz,
          cr::local_seconds{(cr::sys_days{cr::September / 28 / 1986} + 2h + 30min).time_since_epoch()},
          cr::choose::latest};

      assert(zt.get_time_zone() == tz);
      assert(zt.get_sys_time() == cr::sys_days{cr::September / 28 / 1986} + 1h + 30min);
    }
  }

  static_assert(std::constructible_from<cr::zoned_time<cr::seconds, const cr::time_zone*>,
                                        const cr::time_zone*,
                                        cr::local_seconds,
                                        cr::choose>);

  static_assert(!std::constructible_from<cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::none>>,
                                         offset_time_zone<offset_time_zone_flags::none>,
                                         cr::local_seconds,
                                         cr::choose>);

  static_assert(
      !std::constructible_from<cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::has_default_zone>>,
                               offset_time_zone<offset_time_zone_flags::has_default_zone>,
                               cr::local_seconds,
                               cr::choose>);

  static_assert(
      !std::constructible_from<cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::has_locate_zone>>,
                               offset_time_zone<offset_time_zone_flags::has_locate_zone>,
                               cr::local_seconds,
                               cr::choose>);

  static_assert(!std::constructible_from<cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::both>>,
                                         offset_time_zone<offset_time_zone_flags::both>,
                                         cr::local_seconds,
                                         cr::choose>);

  return 0;
}
