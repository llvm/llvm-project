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

// class utc_clock {
// public:
//     using rep                       = a signed arithmetic type;
//     using period                    = ratio<unspecified, unspecified>;
//     using duration                  = chrono::duration<rep, period>;
//     using time_point                = chrono::time_point<utc_clock>;
//     static constexpr bool is_steady = unspecified;
//
//     ...
// };
//
// template<class Duration>
// using utc_time  = time_point<utc_clock, Duration>;
// using utc_seconds = utc_time<seconds>;

#include <concepts>
#include <chrono>
#include <ratio>

#include "test_macros.h"

// class utc_clock
using rep                = std::chrono::utc_clock::rep;
using period             = std::chrono::utc_clock::period;
using duration           = std::chrono::utc_clock::duration;
using time_point         = std::chrono::utc_clock::time_point;
constexpr bool is_steady = std::chrono::utc_clock::is_steady;

// Tests the values. Some of them are implementation-defined.
LIBCPP_STATIC_ASSERT(std::same_as<rep, std::chrono::system_clock::rep>);
static_assert(std::is_arithmetic_v<rep>);
static_assert(std::is_signed_v<rep>);

LIBCPP_STATIC_ASSERT(std::same_as<period, std::chrono::system_clock::period>);
static_assert(std::same_as<period, std::ratio<period::num, period::den>>);

static_assert(std::same_as<duration, std::chrono::duration<rep, period>>);
static_assert(std::same_as<time_point, std::chrono::time_point<std::chrono::utc_clock>>);
LIBCPP_STATIC_ASSERT(is_steady == false);

// typedefs
static_assert(std::same_as<std::chrono::utc_time<int>, std::chrono::time_point<std::chrono::utc_clock, int>>);
static_assert(std::same_as<std::chrono::utc_time<long>, std::chrono::time_point<std::chrono::utc_clock, long>>);
static_assert(std::same_as<std::chrono::utc_seconds, std::chrono::utc_time<std::chrono::seconds>>);
