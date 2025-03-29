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

namespace cr = std::chrono;

// Verify the results of the constructed object.
int main(int, char**) {
  {
    // zoned_time() -> zoned_time<seconds>;
    cr::zoned_time zt;
    static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
  }

  {
    // template<class Duration>
    //   zoned_time(sys_time<Duration>)
    //     -> zoned_time<common_type_t<Duration, seconds>>;
    {
      cr::zoned_time zt{cr::sys_time<cr::nanoseconds>{cr::nanoseconds{0}}};
      static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
    }
    {
      cr::zoned_time zt{cr::sys_time<cr::seconds>{cr::seconds{0}}};
      static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
    }
    {
      cr::zoned_time zt{cr::sys_time<cr::days>{cr::days{0}}};
      static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
    }
  }

  {
    // template<class TimeZonePtrOrName>
    //   zoned_time(TimeZonePtrOrName&&)
    //     -> zoned_time<seconds, time-zone-representation<TimeZonePtrOrName>>;
    { // Name
      {
        cr::zoned_time zt{"UTC"};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{std::string{"UTC"}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{std::string_view{"UTC"}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
    }
    { // TimeZonePtr
      {
        cr::zoned_time zt{static_cast<const cr::time_zone*>(nullptr)};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{offset_time_zone<offset_time_zone_flags::none>{}};
        static_assert(
            std::same_as< decltype(zt), cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::none>>>);
      }
      {
        cr::zoned_time zt{offset_time_zone<offset_time_zone_flags::has_default_zone>{}};
        static_assert(
            std::same_as<decltype(zt),
                         cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::has_default_zone>>>);
      }
      {
        cr::zoned_time zt{offset_time_zone<offset_time_zone_flags::has_locate_zone>{}};
        static_assert(
            std::same_as<decltype(zt),
                         cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::has_locate_zone>>>);
      }
      {
        cr::zoned_time zt{offset_time_zone<offset_time_zone_flags::both>{}};
        static_assert(
            std::same_as< decltype(zt), cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::both>>>);
      }

      // There are no requirements on the TimeZonePtr type.
      {
        cr::zoned_time zt{0};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, int>>);
      }
      {
        cr::zoned_time zt{0.0};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, double>>);
      }
      {
        cr::zoned_time zt{cr::seconds{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, cr::seconds>>);
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
        cr::zoned_time zt{"UTC", cr::sys_time<cr::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{"UTC", cr::sys_time<cr::days>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
    }

    { // TimeZonePtr
      {
        cr::zoned_time zt{static_cast<const cr::time_zone*>(nullptr), cr::sys_time<cr::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{static_cast<const cr::time_zone*>(nullptr), cr::sys_time<cr::days>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{offset_time_zone<offset_time_zone_flags::none>{}, cr::sys_time<cr::nanoseconds>{}};
        static_assert(std::same_as< decltype(zt),
                                    cr::zoned_time<cr::nanoseconds, offset_time_zone<offset_time_zone_flags::none>>>);
      }
      {
        cr::zoned_time zt{offset_time_zone<offset_time_zone_flags::none>{}, cr::sys_time<cr::days>{}};
        static_assert(
            std::same_as< decltype(zt), cr::zoned_time<cr::seconds, offset_time_zone<offset_time_zone_flags::none>>>);
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
        cr::zoned_time zt{"UTC", cr::local_time<cr::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{"UTC", cr::local_time<cr::days>{}, cr::choose::earliest};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
    }
    { // TimeZonePtr
      {
        cr::zoned_time zt{cr::locate_zone("UTC"), cr::local_time<cr::nanoseconds>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{cr::locate_zone("UTC"), cr::local_time<cr::days>{}, cr::choose::earliest};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
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
        cr::zoned_time zt{
            "UTC", cr::zoned_time<cr::nanoseconds, offset_time_zone<offset_time_zone_flags::has_default_zone>>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{"UTC",
                          cr::zoned_time<cr::days, offset_time_zone<offset_time_zone_flags::has_default_zone>>{},
                          cr::choose::earliest};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
    }
    { // TimeZonePtr
      {
        cr::zoned_time zt{
            cr::locate_zone("UTC"),
            cr::zoned_time<cr::nanoseconds, offset_time_zone<offset_time_zone_flags::has_default_zone>>{}};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::nanoseconds, const cr::time_zone*>>);
      }
      {
        cr::zoned_time zt{cr::locate_zone("UTC"),
                          cr::zoned_time<cr::days, offset_time_zone<offset_time_zone_flags::has_default_zone>>{},
                          cr::choose::earliest};
        static_assert(std::same_as<decltype(zt), cr::zoned_time<cr::seconds, const cr::time_zone*>>);
      }
    }
  }

  return 0;
}
