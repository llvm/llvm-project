//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// std::ranges::reserve_hint

#include <cassert>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

using RangeReserveHintT = decltype(std::ranges::reserve_hint);

struct Incomplete;
static_assert(!std::is_invocable_v<RangeReserveHintT, Incomplete[]>);
static_assert(!std::is_invocable_v<RangeReserveHintT, Incomplete (&)[]>);
static_assert(!std::is_invocable_v<RangeReserveHintT, Incomplete (&&)[]>);

extern int bounded_array[42];
extern int unbounded_array[];

struct SizedSentinelRange {
  int data_[42] = {};
  constexpr int* begin() { return data_; }
  constexpr auto end() { return sized_sentinel<int*>(data_ + 42); }
};

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

struct ReserveHintMemberBool {
  constexpr bool reserve_hint() { return false; }
};

static_assert(!std::is_invocable_v<RangeReserveHintT, ReserveHintMemberBool>);

constexpr std::same_as<std::size_t> decltype(auto) bounded_hint = std::ranges::reserve_hint(bounded_array);
static_assert(bounded_hint == 42);

static_assert(!std::is_invocable_v<RangeReserveHintT, decltype(unbounded_array)>);

bool constexpr test_sized_sentinel_range() {
  SizedSentinelRange b;
  std::same_as<std::size_t> decltype(auto) hint = std::ranges::reserve_hint(b);
  assert(hint == 42);

  return true;
}

constexpr std::same_as<std::size_t> decltype(auto) size_member_hint = std::ranges::reserve_hint(SizeMember{});
static_assert(size_member_hint == 42);

constexpr std::same_as<std::size_t> decltype(auto) size_function_hint = std::ranges::reserve_hint(SizeFunction{});
static_assert(size_function_hint == 42);

constexpr std::same_as<std::size_t> decltype(auto) member_hint = std::ranges::reserve_hint(ReserveHintMember{});
static_assert(member_hint == 42);

constexpr std::same_as<std::size_t> decltype(auto) function_hint = std::ranges::reserve_hint(ReserveHintFunction{});
static_assert(function_hint == 42);

// test that the order of preference is ranges::size, then member reserve_hint,
// then function reserve_hint
struct SizeAndReserveHint {
  constexpr std::size_t size() { return 42; }
  constexpr std::size_t reserve_hint() { return 0; }
  friend constexpr std::size_t reserve_hint(SizeAndReserveHint) { return 0; }
};

struct ReserveHintMemberAndFunction {
  constexpr std::size_t reserve_hint() { return 42; }
  friend constexpr std::size_t reserve_hint(ReserveHintMemberAndFunction) { return 0; }
};

static_assert(std::ranges::reserve_hint(SizeAndReserveHint{}) == 42);
static_assert(std::ranges::reserve_hint(ReserveHintMemberAndFunction{}) == 42);

int main(int, char**) {
  test_sized_sentinel_range();
  static_assert(test_sized_sentinel_range());

  return 0;
}
