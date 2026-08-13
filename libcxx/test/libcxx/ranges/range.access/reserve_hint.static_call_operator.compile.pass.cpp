//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Test the libc++-specific behavior that the call operator of the
// std::ranges::reserve_hint customization point object is static.

#include <cstddef>
#include <ranges>

using RangeReserveHintT = decltype(std::ranges::reserve_hint);

extern int bounded_array[42];

struct SizeMember {
  constexpr std::size_t size() { return 42; }
};

struct SizeFunction {
  friend constexpr std::size_t size(SizeFunction) { return 42; }
};

struct ReserveHintMember {
  constexpr std::size_t reserve_hint() { return 42; }
};

struct ReserveHintFunction {
  friend constexpr std::size_t reserve_hint(ReserveHintFunction) { return 42; }
};

static_assert(RangeReserveHintT::operator()(bounded_array) == 42);
static_assert(RangeReserveHintT::operator()(SizeMember{}) == 42);
static_assert(RangeReserveHintT::operator()(SizeFunction{}) == 42);
static_assert(RangeReserveHintT::operator()(ReserveHintMember{}) == 42);
static_assert(RangeReserveHintT::operator()(ReserveHintFunction{}) == 42);
