//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// REQUIRES: has-unix-headers
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <chrono>
//
// class gps_clock;

// static gps_time<common_type_t<_Duration, seconds>>
// from_utc(const utc_time<_Duration>& t) noexcept;

#include <chrono>

#include "check_assertion.h"

// The function is specified as
//   gps_time<common_type_t<Duration, seconds>>{t.time_since_epoch()} + 378691210s
// When t == t.max() there will be a signed integral overflow (other values too).
int main(int, char**) {
  using namespace std::literals::chrono_literals;
  constexpr std::chrono::seconds offset{315964809};

  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::nanoseconds>::max() - offset);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::nanoseconds>::max() - offset + 1ns),
      "the UTC to GPS conversion would overflow");

  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::microseconds>::max() - offset);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::microseconds>::max() - offset + 1us),
      "the UTC to GPS conversion would overflow");

  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::milliseconds>::max() - offset);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::milliseconds>::max() - offset + 1ms),
      "the UTC to GPS conversion would overflow");

  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_seconds::max() - offset);
  TEST_LIBCPP_ASSERT_FAILURE(std::chrono::gps_clock::from_utc(std::chrono::utc_seconds::max() - offset + 1s),
                             "the UTC to GPS conversion would overflow");

  // The conversion uses `common_type_t<Duration, seconds>` so types "larger"
  // than seconds are converted to seconds. Types "larger" than seconds are
  // stored in "smaller" intergral and the overflow can never occur.

  // Validate the types can never overflow on all current (and future) supported platforms.
  static_assert(std::chrono::utc_time<std::chrono::days>::max() <= std::chrono::utc_seconds::max() - offset);
  static_assert(std::chrono::utc_time<std::chrono::weeks>::max() <= std::chrono::utc_seconds::max() - offset);
  static_assert(std::chrono::utc_time<std::chrono::months>::max() <= std::chrono::utc_seconds::max() - offset);
  static_assert(std::chrono::utc_time<std::chrono::years>::max() <= std::chrono::utc_seconds::max() - offset);

  // Validate the run-time conversion works.
  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::days>::max());
  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::weeks>::max());
  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::months>::max());
  (void)std::chrono::gps_clock::from_utc(std::chrono::utc_time<std::chrono::years>::max());

  return 0;
}
