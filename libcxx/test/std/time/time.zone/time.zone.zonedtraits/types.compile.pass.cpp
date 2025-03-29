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

// template<class T> struct zoned_traits {};
//
// A specialization for const time_zone* is provided by the implementation:
// template<> struct zoned_traits<const time_zone*> { ... }

#include <chrono>
#include <type_traits>

// This test test whether non-specialized versions exhibit the expected
// behavior. (Note these specializations are not really useful.)
static_assert(std::is_trivial_v<std::chrono::zoned_traits<int>>);
static_assert(std::is_trivial_v<std::chrono::zoned_traits<float>>);
static_assert(std::is_trivial_v<std::chrono::zoned_traits<void*>>);

struct foo {};
static_assert(std::is_empty_v<std::chrono::zoned_traits<foo>>);
static_assert(std::is_trivial_v<std::chrono::zoned_traits<foo>>);
