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

struct HasSizeMember {
  constexpr std::size_t size() { return 42; }
};

struct HasSizeFunction {
  friend constexpr std::size_t size(HasSizeFunction _) { return 42; }
};

struct HasReserveHintMember {
  constexpr std::size_t reserve_hint() { return 42; }
};

struct HasReserveHintFunction {
  friend constexpr std::size_t reserve_hint(HasReserveHintFunction _) { return 42; }
};

struct HasReserveHintMemberBool {
  constexpr bool reserve_hint() { return false; }
};

static_assert(!std::is_invocable_v<RangeReserveHintT, HasReserveHintMemberBool>);

static_assert(std::ranges::reserve_hint(bounded_array) == 42);
ASSERT_SAME_TYPE(decltype(std::ranges::reserve_hint(bounded_array)), std::size_t);

static_assert(!std::is_invocable_v<RangeReserveHintT, decltype(unbounded_array)>);

bool constexpr test_sized_sentinel_range() {
  SizedSentinelRange b;
  assert(std::ranges::reserve_hint(b) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::reserve_hint(b)), std::size_t);

  return true;
}

static_assert(std::ranges::reserve_hint(HasSizeMember{}) == 42);
ASSERT_SAME_TYPE(decltype(std::ranges::reserve_hint(HasSizeMember{})), std::size_t);

static_assert(std::ranges::reserve_hint(HasSizeFunction{}) == 42);
ASSERT_SAME_TYPE(decltype(std::ranges::reserve_hint(HasSizeFunction{})), std::size_t);

static_assert(std::ranges::reserve_hint(HasReserveHintMember{}) == 42);
ASSERT_SAME_TYPE(decltype(std::ranges::reserve_hint(HasReserveHintMember{})), std::size_t);

static_assert(std::ranges::reserve_hint(HasReserveHintFunction{}) == 42);
ASSERT_SAME_TYPE(decltype(std::ranges::reserve_hint(HasReserveHintFunction{})), std::size_t);

// test that the order of preference is ranges::size, then member reserve_hint,
// then function reserve_hint
struct HasSizeAndReserveHint {
  constexpr std::size_t size() { return 42; }
  constexpr std::size_t reserve_hint() { return 0; }
  friend constexpr std::size_t reserve_hint(HasSizeAndReserveHint _) { return 0; }
};

struct HasReserveHintMemberAndFunction {
  constexpr std::size_t reserve_hint() { return 42; }
  friend constexpr std::size_t reserve_hint(HasReserveHintMemberAndFunction _) { return 0; }
};

static_assert(std::ranges::reserve_hint(HasSizeAndReserveHint{}) == 42);
static_assert(std::ranges::reserve_hint(HasReserveHintMemberAndFunction{}) == 42);

int main(int, char**) {
  test_sized_sentinel_range();
  static_assert(test_sized_sentinel_range());

  return 0;
}
