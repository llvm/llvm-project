//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::empty

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeEmptyT = decltype(std::ranges::empty);

static_assert(!std::is_invocable_v<RangeEmptyT, int[]>);
static_assert(!std::is_invocable_v<RangeEmptyT, int(&)[]>);
static_assert(!std::is_invocable_v<RangeEmptyT, int(&&)[]>);
static_assert( std::is_invocable_v<RangeEmptyT, int[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, const int[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, int (&&)[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, int (&)[1]>);
static_assert( std::is_invocable_v<RangeEmptyT, const int (&)[1]>);

struct Incomplete;
static_assert(!std::is_invocable_v<RangeEmptyT, Incomplete[]>);
static_assert(!std::is_invocable_v<RangeEmptyT, Incomplete(&)[]>);
static_assert(!std::is_invocable_v<RangeEmptyT, Incomplete(&&)[]>);

extern Incomplete array_of_incomplete[42];
static_assert(!std::ranges::empty(array_of_incomplete));
static_assert(!std::ranges::empty(std::move(array_of_incomplete)));
static_assert(!std::ranges::empty(std::as_const(array_of_incomplete)));
static_assert(!std::ranges::empty(static_cast<const Incomplete(&&)[42]>(array_of_incomplete)));

struct InputRangeWithoutSize {
    cpp17_input_iterator<int*> begin() const;
    cpp17_input_iterator<int*> end() const;
};
static_assert(!std::is_invocable_v<RangeEmptyT, const InputRangeWithoutSize&>);

struct NonConstEmpty {
  bool empty();
};
static_assert(!std::is_invocable_v<RangeEmptyT, const NonConstEmpty&>);

struct HasMemberAndFunction {
  constexpr bool empty() const { return true; }
  // We should never do ADL lookup for std::ranges::empty.
  friend bool empty(const HasMemberAndFunction&) { return false; }
};

struct BadReturnType {
  BadReturnType empty() { return {}; }
};
static_assert(!std::is_invocable_v<RangeEmptyT, BadReturnType&>);

struct BoolConvertible {
  constexpr explicit operator bool() noexcept(false) { return true; }
};
struct BoolConvertibleReturnType {
  constexpr BoolConvertible empty() noexcept { return {}; }
};
static_assert(!noexcept(std::ranges::empty(BoolConvertibleReturnType())));

struct InputIterators {
  cpp17_input_iterator<int*> begin() const;
  cpp17_input_iterator<int*> end() const;
};
static_assert(std::is_same_v<decltype(InputIterators().begin() == InputIterators().end()), bool>);
static_assert(!std::is_invocable_v<RangeEmptyT, const InputIterators&>);

constexpr bool testEmptyMember() {
  HasMemberAndFunction a;
  assert(std::ranges::empty(a));

  BoolConvertibleReturnType b;
  assert(std::ranges::empty(b));

  return true;
}

struct SizeMember {
  std::size_t size_;
  constexpr std::size_t size() const { return size_; }
};

struct SizeFunction {
  std::size_t size_;
  friend constexpr std::size_t size(SizeFunction sf) { return sf.size_; }
};

struct BeginEndSizedSentinel {
  constexpr int *begin() const { return nullptr; }
  constexpr auto end() const { return sized_sentinel<int*>(nullptr); }
};
static_assert(std::ranges::forward_range<BeginEndSizedSentinel>);
static_assert(std::ranges::sized_range<BeginEndSizedSentinel>);

constexpr bool testUsingRangesSize() {
  SizeMember a{1};
  assert(!std::ranges::empty(a));
  SizeMember b{0};
  assert(std::ranges::empty(b));

  SizeFunction c{1};
  assert(!std::ranges::empty(c));
  SizeFunction d{0};
  assert(std::ranges::empty(d));

  BeginEndSizedSentinel e;
  assert(std::ranges::empty(e));

  return true;
}

struct BeginEndNotSizedSentinel {
  constexpr int *begin() const { return nullptr; }
  constexpr auto end() const { return sentinel_wrapper<int*>(nullptr); }
};
static_assert( std::ranges::forward_range<BeginEndNotSizedSentinel>);
static_assert(!std::ranges::sized_range<BeginEndNotSizedSentinel>);

// size is disabled here, so we have to compare begin and end.
struct DisabledSizeRangeWithBeginEnd {
  constexpr int *begin() const { return nullptr; }
  constexpr auto end() const { return sentinel_wrapper<int*>(nullptr); }
  std::size_t size() const;
};
template<>
inline constexpr bool std::ranges::disable_sized_range<DisabledSizeRangeWithBeginEnd> = true;
static_assert(std::ranges::contiguous_range<DisabledSizeRangeWithBeginEnd>);
static_assert(!std::ranges::sized_range<DisabledSizeRangeWithBeginEnd>);

struct BeginEndAndEmpty {
  constexpr int *begin() const { return nullptr; }
  constexpr auto end() const { return sentinel_wrapper<int*>(nullptr); }
  constexpr bool empty() { return false; }
};

struct EvilBeginEnd {
  bool empty() &&;
  constexpr int *begin() & { return nullptr; }
  constexpr int *end() & { return nullptr; }
};

constexpr bool testBeginEqualsEnd() {
  BeginEndNotSizedSentinel a;
  assert(std::ranges::empty(a));

  DisabledSizeRangeWithBeginEnd d;
  assert(std::ranges::empty(d));

  BeginEndAndEmpty e;
  assert(!std::ranges::empty(e)); // e.empty()
  assert(std::ranges::empty(std::as_const(e))); // e.begin() == e.end()

  assert(std::ranges::empty(EvilBeginEnd()));

  return true;
}

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeEmptyT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeEmptyT, Holder<Incomplete>*&>);

int main(int, char**) {
  testEmptyMember();
  static_assert(testEmptyMember());

  testUsingRangesSize();
  static_assert(testUsingRangesSize());

  testBeginEqualsEnd();
  static_assert(testBeginEqualsEnd());

  return 0;
}
