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

// static utc_time<common_type_t<_Duration, seconds>>
// to_utc(const gps_time<_Duration>& t) noexcept;

#include <chrono>

#include "check_assertion.h"

// The function is specified as
//   utc_time<common_type_t<Duration, seconds>>{t.time_since_epoch()} - 378691210s
// When t == t.min() there will be a signed integral underlow (other values too).
int main(int, char**) {
  using namespace std::literals::chrono_literals;
  constexpr std::chrono::seconds offset{315964809};

  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::nanoseconds>::min() + offset);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::nanoseconds>::min() + offset - 1ns),
      "the GPS to UTC conversion would underflow");

  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::microseconds>::min() + offset);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::microseconds>::min() + offset - 1us),
      "the GPS to UTC conversion would underflow");

  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::milliseconds>::min() + offset);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::milliseconds>::min() + offset - 1ms),
      "the GPS to UTC conversion would underflow");

  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_seconds::min() + offset);
  TEST_LIBCPP_ASSERT_FAILURE(std::chrono::gps_clock::to_utc(std::chrono::gps_seconds::min() + offset - 1s),
                             "the GPS to UTC conversion would underflow");

  // The conversion uses `common_type_t<Duration, seconds>` so types "larger"
  // than seconds are converted to seconds. Types "larger" than seconds are
  // stored in "smaller" intergral and the underflow can never occur.

  // Validate the types can never underflow on all current (and future) supported platforms.
  static_assert(std::chrono::gps_time<std::chrono::days>::min() >= std::chrono::gps_seconds::min() + offset);
  static_assert(std::chrono::gps_time<std::chrono::weeks>::min() >= std::chrono::gps_seconds::min() + offset);
  static_assert(std::chrono::gps_time<std::chrono::months>::min() >= std::chrono::gps_seconds::min() + offset);
  static_assert(std::chrono::gps_time<std::chrono::years>::min() >= std::chrono::gps_seconds::min() + offset);

  // Validate the run-time conversion works.
  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::days>::min());
  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::weeks>::min());
  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::months>::min());
  (void)std::chrono::gps_clock::to_utc(std::chrono::gps_time<std::chrono::years>::min());

  return 0;
}
