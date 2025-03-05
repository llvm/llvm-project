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

//  class ambiguous_local_time
//
//  template<class Duration>
//    ambiguous_local_time(const local_time<Duration>& tp, const local_info& i);

#include <chrono>

#include "check_assertion.h"

// [time.zone.exception.ambig]/2
//   Preconditions: i.result == local_info::ambiguous is true.
int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      (std::chrono::ambiguous_local_time{
          std::chrono::local_seconds{},
          std::chrono::local_info{-1, //  this is not one of the "named" result values
                                  std::chrono::sys_info{},
                                  std::chrono::sys_info{}}}),
      "creating an ambiguous_local_time from a local_info that is not ambiguous");

  TEST_LIBCPP_ASSERT_FAILURE(
      (std::chrono::ambiguous_local_time{
          std::chrono::local_seconds{},
          std::chrono::local_info{std::chrono::local_info::unique, std::chrono::sys_info{}, std::chrono::sys_info{}}}),
      "creating an ambiguous_local_time from a local_info that is not ambiguous");

  TEST_LIBCPP_ASSERT_FAILURE(
      (std::chrono::ambiguous_local_time{
          std::chrono::local_seconds{},
          std::chrono::local_info{
              std::chrono::local_info::nonexistent, std::chrono::sys_info{}, std::chrono::sys_info{}}}),
      "creating an ambiguous_local_time from a local_info that is not ambiguous");

  return 0;
}
