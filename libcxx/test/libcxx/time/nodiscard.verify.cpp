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

#if TEST_STD_VER >= 20
void test_hh_mm_ss() { // [time.hms]
  const std::chrono::hh_mm_ss<std::chrono::seconds> hms;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hms.is_negative();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hms.hours();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hms.minutes();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hms.seconds();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hms.subseconds();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hms.to_duration();

  using namespace std::chrono_literals;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::is_am(1h);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::is_pm(1h);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::make12(1h);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::chrono::make24(1h, false);
}
#endif

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

#if TEST_STD_VER >= 20
void test_calendar() { // [time.cal]

  { // [time.cal.day]
    const std::chrono::day day{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    day.ok();

    std::chrono::days days;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    day + days;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    days + day;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    day - days;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    day - day;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::day> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(day);
#  endif
  }

  {
    using namespace std::chrono_literals;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94d;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82y;
  }

  { // [time.cal.month]
    const std::chrono::month month{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month.ok();

    std::chrono::months months;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month + months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    months + month;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month - months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month - month;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::month> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(month);
#  endif
  }

  { // [time.cal.mwd]
    std::chrono::month month;
    std::chrono::weekday_indexed wdidx;

    const std::chrono::month_weekday mwd{month, wdidx};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwd.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwd.weekday_indexed();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwd.ok();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month / wdidx;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 / wdidx;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdidx / month;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdidx / 82;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::month_weekday> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(mwd);
#  endif
  }

  { // [time.cal.mwdlast]
    std::chrono::month month;
    std::chrono::weekday_last wdl{{}};

    const std::chrono::month_weekday_last mwdl{month, wdl};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwdl.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwdl.weekday_last();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwdl.ok();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month / wdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 / wdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdl / month;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdl / 82;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::month_weekday_last> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(mwdl);
#  endif
  }

  { // [time.cal.md]
    const std::chrono::month_day md{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    md.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    md.day();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    md.ok();

    std::chrono::month month;
    std::chrono::day day;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month / day;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    day / month;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month / 94;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82 / day;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    day / 49;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::month_day> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(md);
#  endif
  }

  { // [time.cal.mdlast]
    std::chrono::month month;

    const std::chrono::month_day_last mdl{month};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mdl.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mdl.ok();

    std::chrono::last_spec ls;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    month / ls;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ls / month;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 / ls;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ls / 82;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::month_day_last> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(mdl);
#  endif
  }

  { // [time.cal.wd]
    const std::chrono::weekday wd{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd.c_encoding();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd.iso_encoding();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd.ok();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd[0];
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd[std::chrono::last_spec{}];

    std::chrono::days days;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd + days;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    days + wd;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd - days;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wd - wd;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::weekday> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(wd);
#  endif
  }

  { // [time.cal.wdidx]
    const std::chrono::weekday_indexed wdidx{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdidx.weekday();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdidx.index();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdidx.ok();

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::weekday_indexed> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(wdidx);
#  endif
  }

  { // [time.cal.wdlast]
    const std::chrono::weekday_last wdl{std::chrono::weekday{}};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdl.weekday();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wdl.ok();

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::weekday_last> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(wdl);
#  endif
  }

  { // [time.cal.year]
    const std::chrono::year year{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    +year;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    -year;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year.is_leap();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year.ok();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year.max();

    std::chrono::years years;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year + years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    years + year;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year - years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year - year;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::year> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(year);
#  endif
  }

  { // [time.cal.ym]
    const std::chrono::year_month ym{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym.year();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym.month();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym.ok();

    std::chrono::year year;
    std::chrono::month month;
    std::chrono::months months;
    std::chrono::years years;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year / month;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year / 94;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym + months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    months + ym;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym + years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    years + ym;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym - ym;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym - months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym - years;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::year_month> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(ym);
#  endif
  }

  { // [time.cal.ymd]
    const std::chrono::year_month_day ymd{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd.year();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd.day();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd.ok();

    std::chrono::year_month ym;
    std::chrono::day day;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym / day;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym / 94;

    std::chrono::year year;
    std::chrono::month_day md;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year / md;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    82 / md;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    md / year;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    md / 49;

    std::chrono::months months;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd + months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    months + ymd;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd - months;

    std::chrono::years years;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd + years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    years + ymd;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymd - years;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::year_month_day> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(ymd);
#  endif
  }

  { // [time.cal.ymdlast]
    std::chrono::year year;
    std::chrono::month_day_last mdl{{}};

    const std::chrono::year_month_day_last ymdl{year, mdl};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl.year();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl.month_day_last();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl.day();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl.ok();

    std::chrono::year_month ym;
    std::chrono::last_spec ls;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym / ls;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year / mdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 / mdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mdl / year;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mdl / 82;

    std::chrono::months months;
    std::chrono::years years;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl + months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    months + ymdl;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl - months;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl + years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    years + ymdl;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymdl - years;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::year_month_day_last> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(ymdl);
#  endif
  }

  { // [time.cal.ymwd]
    const std::chrono::year_month_weekday ymwd{};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd.year();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd.weekday();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd.index();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd.weekday_indexed();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd.ok();

    std::chrono::year_month ym;
    std::chrono::weekday_indexed wdidx;
    std::chrono::year year;
    std::chrono::month_weekday mw{std::chrono::month{}, wdidx};
    std::chrono::months months;
    std::chrono::years years;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym / wdidx;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year / mw;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 / mw;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mw / year;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mw / 82;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd + months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    months + ymwd;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd - months;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd + years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    years + ymwd;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwd - years;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::year_month_weekday> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(ymwd);
#  endif
  }

  { // [time.cal.ymwdlast]
    std::chrono::year year;
    std::chrono::month month;
    std::chrono::weekday_last wdl{std::chrono::weekday{}};

    const std::chrono::year_month_weekday_last ymwdl{year, month, wdl};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl.year();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl.month();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl.weekday();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl.weekday_last();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl.ok();

    std::chrono::year_month ym;
    std::chrono::month_weekday_last mwdl{std::chrono::month{}, std::chrono::weekday_last{std::chrono::weekday{}}};
    std::chrono::months months;
    std::chrono::years years;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ym / wdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    year / mwdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    94 / mwdl;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwdl / year;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    mwdl / 82;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl + months;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    months + ymwdl;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl - months;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl + years;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    years + ymwdl;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ymwdl - years;

#  if TEST_STD_VER >= 26
    std::hash<std::chrono::year_month_weekday_last> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(ymwdl);
#  endif
  }
}
#endif
