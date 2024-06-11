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

// zoned_time() -> zoned_time<seconds>;
//
// template<class Duration>
//   zoned_time(sys_time<Duration>)
//     -> zoned_time<common_type_t<Duration, seconds>>;
//
// template<class TimeZonePtrOrName>
//   using time-zone-representation =        // exposition only
//     conditional_t<is_convertible_v<TimeZonePtrOrName, string_view>,
//                   const time_zone*,
//                   remove_cvref_t<TimeZonePtrOrName>>;
//
// template<class TimeZonePtrOrName>
//   zoned_time(TimeZonePtrOrName&&)
//     -> zoned_time<seconds, time-zone-representation<TimeZonePtrOrName>>;
//
// template<class TimeZonePtrOrName, class Duration>
//   zoned_time(TimeZonePtrOrName&&, sys_time<Duration>)
//     -> zoned_time<common_type_t<Duration, seconds>,
//                   time-zone-representation<TimeZonePtrOrName>>;
//
// template<class TimeZonePtrOrName, class Duration>
//   zoned_time(TimeZonePtrOrName&&, local_time<Duration>,
//              choose = choose::earliest)
//     -> zoned_time<common_type_t<Duration, seconds>,
//                   time-zone-representation<TimeZonePtrOrName>>;
//
// template<class Duration, class TimeZonePtrOrName, class TimeZonePtr2>
//   zoned_time(TimeZonePtrOrName&&, zoned_time<Duration, TimeZonePtr2>,
//              choose = choose::earliest)
//     -> zoned_time<common_type_t<Duration, seconds>,
//                   time-zone-representation<TimeZonePtrOrName>>;

#include <chrono>
#include <concepts>
#include <string>

#include "test_offset_time_zone.h"

// Verify the results of the constructed object.
int main(int, char**) {
  {
    // zoned_time() -> zoned_time<seconds>;
    std::chrono::zoned_time zt;
    static_assert(
        std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
  }

  {
    // template<class Duration>
    //   zoned_time(sys_time<Duration>)
    //     -> zoned_time<common_type_t<Duration, seconds>>;
    {
      std::chrono::zoned_time zt{std::chrono::sys_time<std::chrono::nanoseconds>{std::chrono::nanoseconds{0}}};
      static_assert(
          std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
    }
    {
      std::chrono::zoned_time zt{std::chrono::sys_time<std::chrono::seconds>{std::chrono::seconds{0}}};
      static_assert(
          std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
    }
    {
      std::chrono::zoned_time zt{std::chrono::sys_time<std::chrono::days>{std::chrono::days{0}}};
      static_assert(
          std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
    }
  }

  {
    // template<class TimeZonePtrOrName>
    //   zoned_time(TimeZonePtrOrName&&)
    //     -> zoned_time<seconds, time-zone-representation<TimeZonePtrOrName>>;
    { // Name
      {
        std::chrono::zoned_time zt{"UTC"};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{std::string{"UTC"}};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{std::string_view{"UTC"}};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
    }
    { // TimeZonePtr
      {
        std::chrono::zoned_time zt{static_cast<const std::chrono::time_zone*>(nullptr)};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{offset_time_zone<offset_time_zone_flags::none>{}};
        static_assert(std::same_as<
                      decltype(zt),
                      std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::none>>>);
      }
      {
        std::chrono::zoned_time zt{offset_time_zone<offset_time_zone_flags::has_default_zone>{}};
        static_assert(
            std::same_as<decltype(zt),
                         std::chrono::zoned_time<std::chrono::seconds,
                                                 offset_time_zone<offset_time_zone_flags::has_default_zone>>>);
      }
      {
        std::chrono::zoned_time zt{offset_time_zone<offset_time_zone_flags::has_locate_zone>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::seconds,
                                                           offset_time_zone<offset_time_zone_flags::has_locate_zone>>>);
      }
      {
        std::chrono::zoned_time zt{offset_time_zone<offset_time_zone_flags::both>{}};
        static_assert(std::same_as<
                      decltype(zt),
                      std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::both>>>);
      }

      // There are no requirements on the TimeZonePtr type.
      {
        std::chrono::zoned_time zt{0};
        static_assert(std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, int>>);
      }
      {
        std::chrono::zoned_time zt{0.0};
        static_assert(std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, double>>);
      }
      {
        std::chrono::zoned_time zt{std::chrono::seconds{}};
        static_assert(std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, std::chrono::seconds>>);
      }
    }
  }

  {
    // template<class TimeZonePtrOrName, class Duration>
    //   zoned_time(TimeZonePtrOrName&&, sys_time<Duration>)
    //     -> zoned_time<common_type_t<Duration, seconds>,
    //                   time-zone-representation<TimeZonePtrOrName>>;
    { // Name
      {
        std::chrono::zoned_time zt{"UTC", std::chrono::sys_time<std::chrono::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{"UTC", std::chrono::sys_time<std::chrono::days>{}};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
    }

    { // TimeZonePtr
      {
        std::chrono::zoned_time zt{
            static_cast<const std::chrono::time_zone*>(nullptr), std::chrono::sys_time<std::chrono::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{
            static_cast<const std::chrono::time_zone*>(nullptr), std::chrono::sys_time<std::chrono::days>{}};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{
            offset_time_zone<offset_time_zone_flags::none>{}, std::chrono::sys_time<std::chrono::nanoseconds>{}};
        static_assert(
            std::same_as<
                decltype(zt),
                std::chrono::zoned_time<std::chrono::nanoseconds, offset_time_zone<offset_time_zone_flags::none>>>);
      }
      {
        std::chrono::zoned_time zt{
            offset_time_zone<offset_time_zone_flags::none>{}, std::chrono::sys_time<std::chrono::days>{}};
        static_assert(std::same_as<
                      decltype(zt),
                      std::chrono::zoned_time<std::chrono::seconds, offset_time_zone<offset_time_zone_flags::none>>>);
      }
    }
  }

  {
    // template<class TimeZonePtrOrName, class Duration>
    //   zoned_time(TimeZonePtrOrName&&, local_time<Duration>,
    //              choose = choose::earliest)
    //     -> zoned_time<common_type_t<Duration, seconds>,
    //                   time-zone-representation<TimeZonePtrOrName>>;
    { // Name
      {
        std::chrono::zoned_time zt{"UTC", std::chrono::local_time<std::chrono::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{"UTC", std::chrono::local_time<std::chrono::days>{}, std::chrono::choose::earliest};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
    }
    { // TimeZonePtr
      {
        std::chrono::zoned_time zt{
            std::chrono::locate_zone("UTC"), std::chrono::local_time<std::chrono::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{
            std::chrono::locate_zone("UTC"),
            std::chrono::local_time<std::chrono::days>{},
            std::chrono::choose::earliest};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
    }
  }

  {
    // template<class Duration, class TimeZonePtrOrName, class TimeZonePtr2>
    //   zoned_time(TimeZonePtrOrName&&, zoned_time<Duration, TimeZonePtr2>,
    //              choose = choose::earliest)
    //     -> zoned_time<common_type_t<Duration, seconds>,
    //                   time-zone-representation<TimeZonePtrOrName>>;
    { // Name
      {
        std::chrono::zoned_time zt{
            "UTC",
            std::chrono::zoned_time<std::chrono::nanoseconds,
                                    offset_time_zone<offset_time_zone_flags::has_default_zone>>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{
            "UTC",
            std::chrono::zoned_time<std::chrono::days, offset_time_zone<offset_time_zone_flags::has_default_zone>>{},
            std::chrono::choose::earliest};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
    }
    { // TimeZonePtr
      {
        std::chrono::zoned_time zt{
            std::chrono::locate_zone("UTC"),
            std::chrono::zoned_time<std::chrono::nanoseconds,
                                    offset_time_zone<offset_time_zone_flags::has_default_zone>>{}};
        static_assert(std::same_as<decltype(zt),
                                   std::chrono::zoned_time<std::chrono::nanoseconds, const std::chrono::time_zone*>>);
      }
      {
        std::chrono::zoned_time zt{
            std::chrono::locate_zone("UTC"),
            std::chrono::zoned_time<std::chrono::days, offset_time_zone<offset_time_zone_flags::has_default_zone>>{},
            std::chrono::choose::earliest};
        static_assert(
            std::same_as<decltype(zt), std::chrono::zoned_time<std::chrono::seconds, const std::chrono::time_zone*>>);
      }
    }
  }

  return 0;
}
