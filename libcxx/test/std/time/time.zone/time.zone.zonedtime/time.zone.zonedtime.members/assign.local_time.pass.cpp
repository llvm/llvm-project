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

#include "test_macros.h"

namespace cr = std::chrono;

// Tests unique conversions. To make sure the test is does not depend on changes
// in the database it uses a time zone with a fixed offset.
static void test_unique() {
  // common_type_t<duration, seconds> -> duration
  {
    using duration         = cr::nanoseconds;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result = zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  {
    using duration         = cr::microseconds;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result = zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  {
    using duration         = cr::milliseconds;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result = zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  // common_type_t<seconds, seconds> -> seconds
  {
    using duration         = cr::seconds;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{42}});
    assert(zt.get_local_time() == local_time_point{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result = zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == sys_time_point{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == local_time_point{duration{99}});
  }
  // common_type_t<duration, seconds> -> seconds
  {
    using duration         = cr::days;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == cr::sys_seconds{duration{42}});
    assert(zt.get_local_time() == cr::local_seconds{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result = zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == cr::sys_seconds{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == cr::local_seconds{duration{99}});
  }
  {
    using duration         = cr::weeks;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == cr::sys_seconds{duration{42}});
    assert(zt.get_local_time() == cr::local_seconds{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result = zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == cr::sys_seconds{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == cr::local_seconds{duration{99}});
  }
  /* This does not work; due to using __tp_ = __zone_->to_sys(__tp);
   * Here the ambiguous/non-existent exception can't stream months and years,
   * leading to a compilation error.
  {
    using duration         = cr::months;
    using sys_time_point   = cr::sys_time<duration>;
    using local_time_point = cr::local_time<duration>;
    using zoned_time       = cr::zoned_time<duration>;
    zoned_time zt{"Etc/GMT+1", sys_time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == cr::sys_seconds{duration{42}});
    assert(zt.get_local_time() == cr::local_seconds{duration{42} - cr::hours{1}});

    std::same_as<zoned_time&> decltype(auto) result= zt = local_time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("Etc/GMT+1"));
    assert(zt.get_sys_time() == cr::sys_seconds{duration{99} + cr::hours{1}});
    assert(zt.get_local_time() == cr::local_seconds{duration{99}});
  } */
}

// Tests non-existent conversions.
static void test_nonexistent() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using namespace std::literals::chrono_literals;

  const cr::time_zone* tz = cr::locate_zone("Europe/Berlin");

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
  cr::local_time<cr::seconds> time{(cr::sys_days{cr::March / 30 / 1986} + 2h + 30min).time_since_epoch()};

  using duration   = cr::seconds;
  using zoned_time = cr::zoned_time<duration>;
  zoned_time zt{tz};

  bool thrown = false;
  try {
    std::same_as<zoned_time&> decltype(auto) result = zt = time;
    assert(&result == &zt);
  } catch (const cr::nonexistent_local_time&) {
    thrown = true;
  }
  // There is no system type that can represent the current local time. So the
  // assertion passes. The current implementation throws an exception too.
  assert(zt.get_local_time() != time);
  assert(thrown);
#endif // TEST_HAS_NO_EXCEPTIONS
}

// Tests ambiguous conversions.
static void test_ambiguous() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using namespace std::literals::chrono_literals;

  const cr::time_zone* tz = cr::locate_zone("Europe/Berlin");

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
  cr::local_time<cr::seconds> time{(cr::sys_days{cr::September / 28 / 1986} + 2h + 30min).time_since_epoch()};

  using duration   = cr::seconds;
  using zoned_time = cr::zoned_time<duration>;
  zoned_time zt{tz};

  bool thrown = false;
  try {
    std::same_as<zoned_time&> decltype(auto) result = zt = time;
    assert(&result == &zt);
  } catch (const cr::ambiguous_local_time&) {
    thrown = true;
  }
  // There is no system type that can represent the current local time. So the
  // assertion passes. The current implementation throws an exception too.
  assert(zt.get_local_time() != time);
  assert(thrown);
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test_unique();
  test_nonexistent();
  test_ambiguous();

  return 0;
}
