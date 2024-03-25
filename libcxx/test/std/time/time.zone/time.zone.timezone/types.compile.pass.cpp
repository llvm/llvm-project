//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class time_zone
// {
//   time_zone(time_zone&&)            = default;
//   time_zone& operator=(time_zone&&) = default;
//
//   ...
// };

#include <chrono>
#include <concepts>
#include <type_traits>

#include "test_macros.h"

// It's impossible to actually obtain a non-const reference to a time_zone, and
// as a result the move constructor can never be exercised in runtime code. We
// still check the property pedantically.
LIBCPP_STATIC_ASSERT(!std::copy_constructible<std::chrono::time_zone>);
LIBCPP_STATIC_ASSERT(!std::is_copy_assignable_v<std::chrono::time_zone>);
static_assert(std::move_constructible<std::chrono::time_zone>);
static_assert(std::is_move_assignable_v<std::chrono::time_zone>);
