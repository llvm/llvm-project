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

struct HasSizeMember {
  constexpr std::size_t size() { return 42; }
};

struct HasSizeFunction {
  friend constexpr std::size_t size(HasSizeFunction) { return 42; }
};

struct HasReserveHintMember {
  constexpr std::size_t reserve_hint() { return 42; }
};

struct HasReserveHintFunction {
  friend constexpr std::size_t reserve_hint(HasReserveHintFunction) { return 42; }
};

static_assert(RangeReserveHintT::operator()(bounded_array) == 42);
static_assert(RangeReserveHintT::operator()(HasSizeMember{}) == 42);
static_assert(RangeReserveHintT::operator()(HasSizeFunction{}) == 42);
static_assert(RangeReserveHintT::operator()(HasReserveHintMember{}) == 42);
static_assert(RangeReserveHintT::operator()(HasReserveHintFunction{}) == 42);
