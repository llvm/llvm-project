//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// file_clock

// check clock invariants

#include <chrono>
#include <type_traits>

#include "test_macros.h"

using C = std::chrono::file_clock;

static_assert((std::is_same<C::rep, C::duration::rep>::value), "");
static_assert((std::is_same<C::period, C::duration::period>::value), "");
static_assert((std::is_same<C::duration, C::time_point::duration>::value), "");
static_assert((std::is_same<C::time_point::clock, C>::value), "");

[[maybe_unused]] constexpr std::same_as<const bool> decltype(auto) _ = C::is_steady;
LIBCPP_STATIC_ASSERT(!C::is_steady);
