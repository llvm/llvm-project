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
// zoned_time& operator=(const local_time<Duration>& st);

// TODO TZDB Investigate the issues in this test, this seems like
// a design issue of the class.
//
// [time.zone.zonedtime.members]/3
//   Effects: After assignment, get_local_time() == lt.
//   This assignment has no effect on the return value of get_time_zone().
//
// The test cases describe the issues.

#include <cassert>
#include <chrono>
#include <concepts>
#include <type_traits>

// Tests unique conversions. To make sure the test is does not depend on changes
// in the database it uses a time zone with a fixed offset.
static void test_unique() {
  // common_type_t<duration, seconds> -> duration
  {
    using duration         = std::chrono::nanoseconds;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  {
    using duration         = std::chrono::microseconds;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  {
    using duration         = std::chrono::milliseconds;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  // common_type_t<seconds, seconds> -> seconds
  {
    using duration         = std::chrono::seconds;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  // common_type_t<duration, seconds> -> seconds
  {
    using duration         = std::chrono::days;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{duration{42}});
    assert(zt.get_local_time() == std::chrono::local_seconds{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == std::chrono::local_seconds{duration{99}});
  }
  {
    using duration         = std::chrono::weeks;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{duration{42}});
    assert(zt.get_local_time() == std::chrono::local_seconds{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == std::chrono::local_seconds{duration{99}});
  }
  /* This does not work; due to using __tp_ = __zone_->to_sys(__tp);
   * Here the ambiguous/non-existent exception can't stream months and years,
   * leading to a compilation error.
  {
    using duration         = std::chrono::months;
    using sys_time_point   = std::chrono::sys_time<duration>;
    using local_time_point = std::chrono::local_time<duration>;
    using zoned_time       = std::chrono::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{duration{42}});
    assert(zt.get_local_time() == std::chrono::local_seconds{duration{42} - std::chrono::hours{1}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = local_time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == std::chrono::sys_seconds{duration{99} + std::chrono::hours{1}});
    assert(zt.get_local_time() == std::chrono::local_seconds{duration{99}});
  } */
}

// Tests non-existant conversions.
static void test_nonexistent() {
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Berlin");

  // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
  // ...
  // 1 DE CE%sT 1980
  // 1 E CE%sT
  //
  // ...
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -

  // Pick an historic date where it's well known what the time zone rules were.
  // This makes it unlikely updates to the database change these rules.
  std::chrono::local_time<std::chrono::seconds> time{
      (std::chrono::sys_days{std::chrono::March / 30 / 1986} + 2h + 30min).time_since_epoch()};

  using duration         = std::chrono::seconds;
  using sys_time_point   = std::chrono::sys_time<duration>;
  using local_time_point = std::chrono::local_time<duration>;
  using zoned_time       = std::chrono::zoned_time<duration>;
  zoned_time zt{tz};

#ifndef TEST_NOEXCEPT
  bool thrown = false;
  try {
    std::same_as<zoned_time&> decltype(auto) _ = zt = time;
  } catch (const std::chrono::nonexistent_local_time&) {
    thrown = true;
  }
  // There is no system type that can represent the current local time. So the
  // assertion passes. The current implementation throws an exception too.
  assert(zt.get_local_time() != time);
  assert(thrown);
#endif
}

// Tests ambiguous conversions.
static void test_ambiguous() {
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Berlin");

  // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
  // ...
  // 1 DE CE%sT 1980
  // 1 E CE%sT
  //
  // ...
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -

  // Pick an historic date where it's well known what the time zone rules were.
  // This makes it unlikely updates to the database change these rules.
  std::chrono::local_time<std::chrono::seconds> time{
      (std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h + 30min).time_since_epoch()};

  using duration         = std::chrono::seconds;
  using sys_time_point   = std::chrono::sys_time<duration>;
  using local_time_point = std::chrono::local_time<duration>;
  using zoned_time       = std::chrono::zoned_time<duration>;
  zoned_time zt{tz};

#ifndef TEST_NOEXCEPT
  bool thrown = false;
  try {
    std::same_as<zoned_time&> decltype(auto) _ = zt = time;
  } catch (const std::chrono::ambiguous_local_time&) {
    thrown = true;
  }
  // There is no system type that can represent the current local time. So the
  // assertion passes. The current implementation throws an exception too.
  assert(zt.get_local_time() != time);
  assert(thrown);
#endif
}

int main(int, char**) {
  test_unique();
  test_nonexistent();
  test_ambiguous();

  return 0;
}
