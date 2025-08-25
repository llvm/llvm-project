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

// class time_zone;

// template <class _Duration>
//   sys_time<common_type_t<Duration, seconds>>
//     to_sys(const local_time<Duration>& tp, choose z) const;

#include <chrono>
#include <format>
#include <cassert>
#include <string_view>

#include "test_macros.h"

// Tests unique conversions. To make sure the test is does not depend on changes
// in the database it uses a time zone with a fixed offset.
static void test_unique() {
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Etc/GMT+1");

  assert(tz->to_sys(std::chrono::local_time<std::chrono::nanoseconds>{-1ns}, std::chrono::choose::earliest) ==
         std::chrono::sys_time<std::chrono::nanoseconds>{-1ns + 1h});

  assert(tz->to_sys(std::chrono::local_time<std::chrono::microseconds>{0us}, std::chrono::choose::latest) ==
         std::chrono::sys_time<std::chrono::microseconds>{1h});

  assert(tz->to_sys(
             std::chrono::local_time<std::chrono::seconds>{
                 (std::chrono::sys_days{std::chrono::January / 1 / -21970}).time_since_epoch()},
             std::chrono::choose::earliest) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / -21970}).time_since_epoch() + 1h});

  // sys_time<common_type_t<Duration, seconds>> is seconds for the larger types
  assert(tz->to_sys(
             std::chrono::local_time<std::chrono::days>{
                 (std::chrono::sys_days{std::chrono::January / 1 / 21970}).time_since_epoch()},
             std::chrono::choose::latest) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 21970}).time_since_epoch() + 1h});

  assert(tz->to_sys(std::chrono::local_time<std::chrono::weeks>{}, std::chrono::choose::earliest) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() + 1h});

  // Note months and years cannot be streamed; however these functions don't
  // throw an exception and thus can be used.
  assert(tz->to_sys(std::chrono::local_time<std::chrono::months>{}, std::chrono::choose::latest) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() + 1h});

  assert(tz->to_sys(std::chrono::local_time<std::chrono::years>{}, std::chrono::choose::earliest) ==
         std::chrono::sys_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() + 1h});
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
      (std::chrono::sys_days{std::chrono::March / 30 / 1986} + 2h).time_since_epoch()};

  std::chrono::sys_seconds expected{time.time_since_epoch() - 1h};

  // Validates whether the database did not change.
  std::chrono::local_info info = tz->get_info(time);
  assert(info.result == std::chrono::local_info::nonexistent);

  assert(tz->to_sys(time + 0ns, std::chrono::choose::earliest) == expected);
  assert(tz->to_sys(time + 0us, std::chrono::choose::latest) == expected);
  assert(tz->to_sys(time + 0ms, std::chrono::choose::earliest) == expected);
  assert(tz->to_sys(time + 0s, std::chrono::choose::latest) == expected);

  // The entire nonexisting hour should map to the same time.
  // For nonexistant the value of std::chrono::choose has no effect.
  assert(tz->to_sys(time + 1s, std::chrono::choose::earliest) == expected);
  assert(tz->to_sys(time + 1min, std::chrono::choose::latest) == expected);
  assert(tz->to_sys(time + 30min, std::chrono::choose::earliest) == expected);
  assert(tz->to_sys(time + 59min + 59s, std::chrono::choose::latest) == expected);
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
      (std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h).time_since_epoch()};

  std::chrono::sys_seconds earlier{time.time_since_epoch() - 2h};
  std::chrono::sys_seconds later{time.time_since_epoch() - 1h};

  // Validates whether the database did not change.
  std::chrono::local_info info = tz->get_info(time);
  assert(info.result == std::chrono::local_info::ambiguous);

  assert(tz->to_sys(time + 0ns, std::chrono::choose::earliest) == earlier);
  assert(tz->to_sys(time + 0us, std::chrono::choose::latest) == later);
  assert(tz->to_sys(time + 0ms, std::chrono::choose::earliest) == earlier);
  assert(tz->to_sys(time + 0s, std::chrono::choose::latest) == later);

  // Test times in the ambiguous hour
  assert(tz->to_sys(time + 1s, std::chrono::choose::earliest) == earlier + 1s);
  assert(tz->to_sys(time + 1min, std::chrono::choose::latest) == later + 1min);
  assert(tz->to_sys(time + 30min, std::chrono::choose::earliest) == earlier + 30min);
  assert(tz->to_sys(time + 59min + 59s, std::chrono::choose::latest) == later + 59min + 59s);
}

// This test does the basic validations of this function. The library function
// uses `local_info get_info(const local_time<Duration>& tp)` as implementation
// detail. The get_info function does extensive testing of the data.
int main(int, char**) {
  test_unique();
  test_nonexistent();
  test_ambiguous();

  return 0;
}
