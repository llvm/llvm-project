//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// XFAIL: libcpp-has-no-experimental-tzdb

// <chrono>

// template <class _Duration>
//   sys_time<common_type_t<Duration, seconds>>
//     to_sys(const local_time<Duration>& tp, choose z) const;

#include <chrono>

#include "check_assertion.h"

// Tests values that cannot be converted. To make sure the test is does not depend on changes
// in the database it uses a time zone with a fixed offset.
int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::locate_zone("Etc/GMT-1")->to_sys(std::chrono::local_seconds::min(), std::chrono::choose::earliest),
      "cannot convert the local time; it would be before the minimum system clock value");

  // TODO TZDB look why std::chrono::local_seconds::max()  fails
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::locate_zone("Etc/GMT+1")
          ->to_sys(std::chrono::local_seconds::max() - std::chrono::seconds(1), std::chrono::choose::latest),
      "cannot convert the local time; it would be after the maximum system clock value");

  return 0;
}
