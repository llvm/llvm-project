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

// <chrono>

// class gps_clock {
// public:
//     using rep                       = a signed arithmetic type;
//     using period                    = ratio<unspecified, unspecified>;
//     using duration                  = chrono::duration<rep, period>;
//     using time_point                = chrono::time_point<gps_clock>;
//     static constexpr bool is_steady = unspecified;
//
//     ...
// };
//
// template<class Duration>
// using gps_time  = time_point<gps_clock, Duration>;
// using gps_seconds = gps_time<seconds>;

#include <chrono>
#include <concepts>
#include <ratio>

#include "test_macros.h"

// class gps_clock
using rep                = std::chrono::gps_clock::rep;
using period             = std::chrono::gps_clock::period;
using duration           = std::chrono::gps_clock::duration;
using time_point         = std::chrono::gps_clock::time_point;
constexpr bool is_steady = std::chrono::gps_clock::is_steady;

// Tests the values. part of them are implementation defined.
LIBCPP_STATIC_ASSERT(std::same_as<rep, std::chrono::utc_clock::rep>);
static_assert(std::is_arithmetic_v<rep>);
static_assert(std::is_signed_v<rep>);

LIBCPP_STATIC_ASSERT(std::same_as<period, std::chrono::utc_clock::period>);
static_assert(std::same_as<period, std::ratio<period::num, period::den>>);

static_assert(std::same_as<duration, std::chrono::duration<rep, period>>);
static_assert(std::same_as<time_point, std::chrono::time_point<std::chrono::gps_clock>>);
LIBCPP_STATIC_ASSERT(is_steady == false);

// typedefs
static_assert(std::same_as<std::chrono::gps_time<int>, std::chrono::time_point<std::chrono::gps_clock, int>>);
static_assert(std::same_as<std::chrono::gps_time<long>, std::chrono::time_point<std::chrono::gps_clock, long>>);
static_assert(std::same_as<std::chrono::gps_seconds, std::chrono::gps_time<std::chrono::seconds>>);
