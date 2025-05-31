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

//  template<class Duration, class TimeZonePtr = const time_zone*>
//  class zoned_time {
//  public:
//    using duration = common_type_t<Duration, seconds>;
//  ...
//    zoned_time(const zoned_time&) = default;
//    zoned_time& operator=(const zoned_time&) = default;
//
//  };
//
// using zoned_seconds = zoned_time<seconds>;

#include <chrono>
#include <concepts>
#include <type_traits>
#include <memory>

// Test the default template argument
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::days>,
                           std::chrono::zoned_time<std::chrono::days, const std::chrono::time_zone*>>);

// Test duration
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::nanoseconds>::duration, std::chrono::nanoseconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::microseconds>::duration, std::chrono::microseconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::milliseconds>::duration, std::chrono::milliseconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::seconds>::duration, std::chrono::seconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::days>::duration, std::chrono::seconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::weeks>::duration, std::chrono::seconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::months>::duration, std::chrono::seconds>);
static_assert(std::same_as<std::chrono::zoned_time<std::chrono::years>::duration, std::chrono::seconds>);

// Tests defaulted copy construct/assign and move construct/assign
static_assert(std::is_copy_constructible_v<std::chrono::zoned_time<std::chrono::days>>);
static_assert(std::is_move_constructible_v<std::chrono::zoned_time<std::chrono::days>>);
static_assert(std::is_copy_assignable_v<std::chrono::zoned_time<std::chrono::days>>);
static_assert(std::is_move_assignable_v<std::chrono::zoned_time<std::chrono::days>>);

// There are no requirements for TimeZonePtr, so test with a non-pointer type.
static_assert(std::is_copy_constructible_v<std::chrono::zoned_time<std::chrono::days, int>>);
static_assert(std::is_move_constructible_v<std::chrono::zoned_time<std::chrono::days, int>>);
static_assert(std::is_copy_assignable_v<std::chrono::zoned_time<std::chrono::days, int>>);
static_assert(std::is_move_assignable_v<std::chrono::zoned_time<std::chrono::days, int>>);

// Test with a move only type, since the copy constructor is defined, no move
// constuctor is generated.
static_assert(!std::is_copy_constructible_v<std::chrono::zoned_time< std::chrono::days, std::unique_ptr<int>>>);
static_assert(!std::is_move_constructible_v<std::chrono::zoned_time< std::chrono::days, std::unique_ptr<int>>>);
static_assert(!std::is_copy_assignable_v<std::chrono::zoned_time< std::chrono::days, std::unique_ptr<int>>>);
static_assert(!std::is_move_assignable_v<std::chrono::zoned_time< std::chrono::days, std::unique_ptr<int>>>);

// using zoned_seconds = zoned_time<seconds>;
static_assert(std::same_as<std::chrono::zoned_seconds, std::chrono::zoned_time<std::chrono::seconds>>);
