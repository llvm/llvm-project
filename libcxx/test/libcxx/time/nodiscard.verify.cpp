//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// Check that functions are marked [[nodiscard]]

#include <chrono>
#include <ctime>
#include <ratio>

#include "test_macros.h"

#if TEST_STD_VER >= 20
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

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::leap_second> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(leap);
#  endif
  }

  {
    using t = std::chrono::zoned_traits<const std::chrono::time_zone*>;
    t::default_zone();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    t::locate_zone(""); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
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

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::zoned_time<std::chrono::seconds>> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(zt);
#  endif
  }
}
#endif // TEST_STD_VER >= 20

void test_duration() { // [time.duration]
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

#if TEST_STD_VER >= 17
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
#endif // TEST_STD_VER >= 14

#if TEST_STD_VER >= 26
  std::hash<std::chrono::duration<int, std::ratio<1, 30>>> hash;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hash(dr);
#endif
}

void test_time_point() { // [time.point]
  std::chrono::time_point<std::chrono::system_clock> tp;
  std::chrono::duration<double, std::ratio<1, 30> > dr;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  tp.time_since_epoch();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  tp.min();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  tp.max();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::time_point_cast<std::chrono::seconds>(tp);

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::floor<std::chrono::seconds>(tp);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::ceil<std::chrono::seconds>(tp);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::round<std::chrono::seconds>(tp);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::abs(dr);
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  tp + dr;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  dr + tp;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  tp - dr;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  tp - tp;
}

void test_clocks() { // [time.clock]
#if TEST_STD_VER >= 20
  { // [time.clock.file]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::file_clock::now();

    using Duration = std::chrono::duration<double, std::ratio<1, 30>>;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::file_clock::to_sys(std::chrono::file_time<Duration>{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::file_clock::from_sys(std::chrono::sys_time<Duration>{});
  }
#endif

#if TEST_STD_VER >= 20
  { // [time.clock.gps]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::gps_clock::now();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::gps_clock::to_utc(std::chrono::gps_seconds{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::gps_clock::from_utc(std::chrono::utc_seconds{});
  }
#endif // TEST_STD_VER >= 20

#if _LIBCPP_HAS_MONOTONIC_CLOCK
  { // [time.clock.steady]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::steady_clock::now();
  }
#endif

  { // [time.clock.system]

    std::chrono::time_point<std::chrono::system_clock> tp;
    std::time_t time = std::time(nullptr);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::system_clock::now();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::system_clock::to_time_t(tp);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::system_clock::from_time_t(time);

#if TEST_STD_VER >= 26
    std::hash<std::chrono::time_point<std::chrono::system_clock>> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(tp);
#endif
  }

#if TEST_STD_VER >= 20
  { // [time.clock.tai]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::tai_clock::now();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::tai_clock::to_utc(std::chrono::tai_seconds{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::chrono::tai_clock::from_utc(std::chrono::utc_seconds{});
  }
#endif

#if TEST_STD_VER >= 20
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
#endif // TEST_STD_VER >= 20
}
