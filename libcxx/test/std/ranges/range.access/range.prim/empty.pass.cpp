//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// std::ranges::empty

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

using RangeEmptyT = decltype(std::ranges::empty);
using RangeSizeT = decltype(std::ranges::size);

static_assert(!std::is_invocable_v<RangeEmptyT, int[]>);
static_assert(!std::is_invocable_v<RangeEmptyT, int(&)[]>);
static_assert(!std::is_invocable_v<RangeEmptyT, int(&&)[]>);
static_assert( std::is_invocable_v<RangeEmptyT, int[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, const int[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, int (&&)[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, int (&)[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, const int (&)[1]>);

struct NonConstSizeAndEmpty {
  int size();
  bool empty();
};
static_assert(!std::is_invocable_v<RangeSizeT, const NonConstSizeAndEmpty&>);
static_assert(!std::is_invocable_v<RangeEmptyT, const NonConstSizeAndEmpty&>);

struct HasMemberAndFunction {
  constexpr bool empty() const { return true; }
  // We should never do ADL lookup for std::ranges::empty.
  friend bool empty(const HasMemberAndFunction&) { return false; }
};

struct BadReturnType {
  BadReturnType empty() { return {}; }
};
static_assert(!std::is_invocable_v<RangeEmptyT, BadReturnType>);

struct BoolConvertible {
  constexpr operator bool() noexcept(false) { return true; }
};
struct BoolConvertibleReturnType {
  constexpr BoolConvertible empty() noexcept { return {}; }
};

static_assert(!noexcept(std::ranges::empty(BoolConvertibleReturnType())));

constexpr bool testEmptyMember() {
  HasMemberAndFunction a;
  assert(std::ranges::empty(a) == true);

  BoolConvertibleReturnType b;
  assert(std::ranges::empty(b) == true);

  return true;
}

struct SizeMember {
  size_t size_;
  constexpr size_t size() const { return size_; }
};

struct SizeFunction {
  size_t size_;
  friend constexpr size_t size(SizeFunction sf) { return sf.size_; }
};

constexpr bool testUsingRangesSize() {
  SizeMember a{1};
  assert(std::ranges::empty(a) == false);
  SizeMember b{0};
  assert(std::ranges::empty(b) == true);

  SizeFunction c{1};
  assert(std::ranges::empty(c) == false);
  SizeFunction d{0};
  assert(std::ranges::empty(d) == true);

  return true;
}

struct other_forward_iterator : forward_iterator<int*> { };

struct sentinel {
  constexpr bool operator==(std::input_or_output_iterator auto) const { return true; }
};

struct BeginEndNotSizedSentinel {
  friend constexpr forward_iterator<int*> begin(BeginEndNotSizedSentinel) { return {}; }
  friend constexpr sentinel end(BeginEndNotSizedSentinel) { return {}; }
};
static_assert(!std::is_invocable_v<RangeSizeT, BeginEndNotSizedSentinel&>);

struct InvalidMinusBeginEnd {
  friend constexpr random_access_iterator<int*> begin(InvalidMinusBeginEnd) { return {}; }
  friend constexpr sentinel end(InvalidMinusBeginEnd) { return {}; }
};

// Int is integer-like, but it is not other_forward_iterator's difference_type.
constexpr short operator-(sentinel, random_access_iterator<int*>) { return 2; }
constexpr short operator-(random_access_iterator<int*>, sentinel) { return 2; }
static_assert(!std::is_invocable_v<RangeSizeT, InvalidMinusBeginEnd&>);

// This type will use ranges::size.
struct IntPtrBeginAndEnd {
  int buff[8];
  constexpr int* begin() { return buff; }
  constexpr int* end() { return buff + 8; }
};
static_assert(std::is_invocable_v<RangeSizeT, IntPtrBeginAndEnd&>);

// size is disabled here, and it isn't sized_sentinel_for, so we have to compare begin
// and end again.
struct DisabledSizeRangeWithBeginEnd {
  friend constexpr forward_iterator<int*> begin(DisabledSizeRangeWithBeginEnd) { return {}; }
  friend constexpr sentinel end(DisabledSizeRangeWithBeginEnd) { return {}; }
  constexpr size_t size() const { return 1; }
};

template <>
inline constexpr bool std::ranges::disable_sized_range<DisabledSizeRangeWithBeginEnd> = true;
static_assert(!std::is_invocable_v<RangeSizeT, DisabledSizeRangeWithBeginEnd&>);

struct BeginEndAndEmpty {
  int* begin();
  int* end();
  constexpr bool empty() { return true; }
};

struct BeginEndAndConstEmpty {
  int* begin();
  int* end();
  constexpr bool empty() const { return true; }
};

constexpr bool testBeginEqualsEnd() {
  BeginEndNotSizedSentinel a;
  assert(std::ranges::empty(a) == true);

  InvalidMinusBeginEnd b;
  assert(std::ranges::empty(b) == true);

  IntPtrBeginAndEnd c;
  assert(std::ranges::empty(c) == false);

  DisabledSizeRangeWithBeginEnd d;
  assert(std::ranges::empty(d) == true);

  BeginEndAndEmpty e;
  assert(std::ranges::empty(e) == true);

  BeginEndAndConstEmpty f;
  assert(std::ranges::empty(f) == true);
  assert(std::ranges::empty(std::as_const(f)) == true);

  return true;
}

int main(int, char**) {
  testEmptyMember();
  static_assert(testEmptyMember());

  testUsingRangesSize();
  static_assert(testUsingRangesSize());

  testBeginEqualsEnd();
  static_assert(testBeginEqualsEnd());

  return 0;
}
