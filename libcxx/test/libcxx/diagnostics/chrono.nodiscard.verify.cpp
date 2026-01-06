//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that format functions are marked [[nodiscard]] as a conforming extension

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

#include <chrono>
#include <ratio>

#include "test_macros.h"

// These types have "private" constructors.
void test(std::chrono::time_zone tz, std::chrono::time_zone_link link, std::chrono::leap_second leap) {
  std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
  list.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.cbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  list.cend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  {
    const std::chrono::tzdb& t = list.front();
    t.locate_zone("name"); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    t.current_zone();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  namespace crno = std::chrono;
  crno::get_tzdb_list();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  crno::get_tzdb();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  crno::locate_zone("n"); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  crno::current_zone();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  crno::remote_version(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  {
    std::chrono::sys_seconds s{};
    std::chrono::local_seconds l{};
    std::chrono::choose z = std::chrono::choose::earliest;
    tz.name();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    tz.get_info(s);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    tz.get_info(l);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    tz.to_sys(l);        // not nodiscard
    tz.to_sys(l, z);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    operator==(tz, tz);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    operator<=>(tz, tz); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    link.name();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    link.target(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    operator==(link, link);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    operator<=>(link, link);
  }

  {
    leap.date();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    leap.value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    using t = std::chrono::zoned_traits<const std::chrono::time_zone*>;
    t::default_zone();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    t::locate_zone(""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  { // [time.clock.utc]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::utc_clock::now();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::utc_clock::to_sys(std::chrono::utc_seconds{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::utc_clock::from_sys(std::chrono::sys_seconds{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::get_leap_second_info(std::chrono::utc_seconds{});
  }

  {
    std::chrono::zoned_time<std::chrono::seconds> zt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    static_cast<std::chrono::sys_seconds>(zt);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    static_cast<std::chrono::local_seconds>(zt);

    zt.get_time_zone();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    zt.get_local_time(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    zt.get_sys_time();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    zt.get_info();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  { // [time.clock.tai]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::tai_clock::now();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::tai_clock::to_utc(std::chrono::tai_seconds{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::tai_clock::from_utc(std::chrono::utc_seconds{});
  }

  { // [time.clock.gps]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::gps_clock::now();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::gps_clock::to_utc(std::chrono::gps_seconds{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::gps_clock::from_utc(std::chrono::utc_seconds{});
  }

  { // [time.duration]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::microseconds(2));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration_values<int>::zero();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration_values<int>::max();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration_values<int>::min();

    std::chrono::duration<int, std::ratio<1, 30> > dr;

#if TEST_STD_VER >= 17
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::floor<std::chrono::seconds>(dr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::ceil<std::chrono::seconds>(dr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::round<std::chrono::seconds>(dr);
#endif

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr.count();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    +dr;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    -dr;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration<int, std::ratio<1, 30> >::zero();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration<int, std::ratio<1, 30> >::max();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::duration<int, std::ratio<1, 30> >::min();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr + dr;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr - dr;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr * 94;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 * dr;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr / 82;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr / dr;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr % 47;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    dr % dr;

    using namespace std::chrono_literals;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94h;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82.5h;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94min;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82.5min;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94s;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82.5s;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94ms;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82.5ms;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94us;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82.5us;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94ns;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82.5ns;
  }
}
