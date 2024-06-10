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
#include <concepts>
#include <cassert>

#include "test_offset_time_zone.h"

int main(int, char**) {
  // Tests unique conversions. To make sure the test is does not depend on changes
  // in the database it uses a time zone with a fixed offset.
  {
    using ptr = const std::chrono::time_zone*;
    ptr tz    = std::chrono::locate_zone("Etc/GMT+1");
    std::chrono::zoned_time<std::chrono::seconds> zt{
        tz, std::chrono::local_seconds{std::chrono::seconds{0}}, std::chrono::choose::earliest};

    assert(zt.get_time_zone() == tz);
    assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::hours{1}});
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
    using ptr = const std::chrono::time_zone*;
    ptr tz    = std::chrono::locate_zone("Europe/Berlin");
    {
      std::chrono::zoned_time<std::chrono::seconds> zt{
          tz,
          std::chrono::local_seconds{
              (std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h + 30min).time_since_epoch()},
          std::chrono::choose::earliest};

      assert(zt.get_time_zone() == tz);
      assert(zt.get_sys_time() == std::chrono::sys_days{std::chrono::September / 28 / 1986} + 0h + 30min);
    }
    {
      std::chrono::zoned_time<std::chrono::seconds> zt{
          tz,
          std::chrono::local_seconds{
              (std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h + 30min).time_since_epoch()},
          std::chrono::choose::latest};

      assert(zt.get_time_zone() == tz);
      assert(zt.get_sys_time() == std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h + 30min);
    }
  }

  static_assert(std::constructible_from<std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>,
                                        const std::chrono::time_zone*,
                                        std::chrono::local_seconds,
                                        std::chrono::choose>);

  static_assert(!std::constructible_from<
                std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::none>>,
                offset_time_zone<offset_time_zone_flags::none>,
                std::chrono::local_seconds,
                std::chrono::choose>);

  static_assert(
      !std::constructible_from<
          std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::has_default_zone>>,
          offset_time_zone<offset_time_zone_flags::has_default_zone>,
          std::chrono::local_seconds,
          std::chrono::choose>);

  static_assert(
      !std::constructible_from<
          std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::has_locate_zone>>,
          offset_time_zone<offset_time_zone_flags::has_locate_zone>,
          std::chrono::local_seconds,
          std::chrono::choose>);

  static_assert(!std::constructible_from<
                std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::both>>,
                offset_time_zone<offset_time_zone_flags::both>,
                std::chrono::local_seconds,
                std::chrono::choose>);

  return 0;
}
