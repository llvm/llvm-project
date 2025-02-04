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
// local_time<common_type_t<Duration, seconds>>
//   to_local(const sys_time<Duration>& tp) const;

#include <chrono>
#include <format>
#include <cassert>
#include <string_view>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

int main(int, char**) {
  // To make sure the test does not depend on changes in the database it uses a
  // time zone with a fixed offset.
  using namespace std::literals::chrono_literals;

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Etc/GMT+1");

  assert(tz->to_local(std::chrono::sys_time<std::chrono::nanoseconds>{-1ns}) ==
         std::chrono::local_time<std::chrono::nanoseconds>{-1ns - 1h});

  assert(tz->to_local(std::chrono::sys_time<std::chrono::microseconds>{0us}) ==
         std::chrono::local_time<std::chrono::microseconds>{0us - 1h});

  assert(tz->to_local(
             std::chrono::sys_time<std::chrono::seconds>{std::chrono::sys_days{std::chrono::January / 1 / -21970}}) ==
         std::chrono::local_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / -21970}).time_since_epoch() - 1h});

  assert(
      tz->to_local(std::chrono::sys_time<std::chrono::days>{std::chrono::sys_days{std::chrono::January / 1 / 21970}}) ==
      std::chrono::local_time<std::chrono::seconds>{
          (std::chrono::sys_days{std::chrono::January / 1 / 21970}).time_since_epoch() - 1h});

  assert(tz->to_local(std::chrono::sys_time<std::chrono::weeks>{}) ==
         std::chrono::local_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() - 1h});

  assert(tz->to_local(std::chrono::sys_time<std::chrono::months>{}) ==
         std::chrono::local_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() - 1h});

  assert(tz->to_local(std::chrono::sys_time<std::chrono::years>{}) ==
         std::chrono::local_time<std::chrono::seconds>{
             (std::chrono::sys_days{std::chrono::January / 1 / 1970}).time_since_epoch() - 1h});

  return 0;
}
