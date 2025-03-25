//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::cbegin
// std::ranges::crbegin

#include <ranges>

#include <cassert>
#include <utility>
#include "almost_satisfies_types.h"
#include "test_macros.h"
#include "test_iterators.h"

using RangeCBeginT  = decltype(std::ranges::cbegin);
using RangeCRBeginT = decltype(std::ranges::crbegin);

static_assert(!std::is_invocable_v<RangeCBeginT, int (&&)[10]>);
static_assert(std::is_invocable_v<RangeCBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&&)[10]>);
static_assert(std::is_invocable_v<RangeCRBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&)[]>);

static_assert(!std::is_invocable_v<RangeCBeginT, InputRangeNotDerivedFrom>);
static_assert(!std::is_invocable_v<RangeCBeginT, InputRangeNotIndirectlyReadable>);
static_assert(!std::is_invocable_v<RangeCBeginT, InputRangeNotInputOrOutputIterator>);
static_assert(!std::is_invocable_v<RangeCBeginT, InputRangeNotSentinelSemiregular>);
static_assert(!std::is_invocable_v<RangeCBeginT, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!std::is_invocable_v<RangeCRBeginT, BidirectionalRangeNotDerivedFrom>);
static_assert(!std::is_invocable_v<RangeCRBeginT, BidirectionalRangeNotSentinelSemiregular>);
static_assert(!std::is_invocable_v<RangeCRBeginT, BidirectionalRangeNotSentinelWeaklyEqualityComparableWith>);
static_assert(!std::is_invocable_v<RangeCRBeginT, BidirectionalRangeNotDecrementable>);

struct Incomplete;

static_assert(!std::is_invocable_v<RangeCBeginT, Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, const Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, Incomplete (&&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, const Incomplete (&&)[10]>);

static_assert(!std::is_invocable_v<RangeCRBeginT, Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, const Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Incomplete (&&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, const Incomplete (&&)[10]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, Incomplete (&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, const Incomplete (&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, Incomplete (&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, const Incomplete (&)[]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, Incomplete (&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, const Incomplete (&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, Incomplete (&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, const Incomplete (&)[10]>);

struct NonborrowingRange {
  int x;
  constexpr const int* begin() const { return &x; }
  constexpr const int* rbegin() const { return &x; }
  constexpr const int* end() const { return &x; }
  constexpr const int* rend() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert(std::is_invocable_v<RangeCBeginT, NonborrowingRange&>);
static_assert(!std::is_invocable_v<RangeCBeginT, NonborrowingRange&&>);
static_assert(std::is_invocable_v<RangeCBeginT, NonborrowingRange const&>);
static_assert(!std::is_invocable_v<RangeCBeginT, NonborrowingRange const&&>);
static_assert(std::is_invocable_v<RangeCRBeginT, NonborrowingRange&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, NonborrowingRange&&>);
static_assert(std::is_invocable_v<RangeCRBeginT, NonborrowingRange const&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, NonborrowingRange const&&>);

constexpr bool testReturnTypes() {
  int* a[2];
  int b[2][2];
  struct PossiblyConstRange {
    char*& begin();
    char*& end();
    const short*& begin() const;
    const short*& end() const;
    int*& rbegin();
    int*& rend();
    const long*& rbegin() const;
    const long*& rend() const;
  } c;
  struct AlwaysConstRange {
    const char*& begin();
    const char*& end();
    const short*& begin() const;
    const short*& end() const;
    const int*& rbegin();
    const int*& rend();
    const long*& rbegin() const;
    const long*& rend() const;
  } d;
  struct NeverConstRange {
    char*& begin();
    char*& end();
    short*& begin() const;
    short& end() const;
    int*& rbegin();
    int*& rend();
    long*& rbegin() const;
    long*& rend() const;
  } e;

  static_assert(!std::ranges::constant_range<PossiblyConstRange>);
  static_assert(std::ranges::constant_range<const PossiblyConstRange>);
  static_assert(std::ranges::constant_range<AlwaysConstRange>);
  static_assert(std::ranges::constant_range<const AlwaysConstRange>);
  static_assert(!std::ranges::constant_range<NeverConstRange>);
  static_assert(!std::ranges::constant_range<const NeverConstRange>);

  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(a)), int* const*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(b)), const int(*)[2]);
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(c)), const short*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(d)), const short*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(e)), std::basic_const_iterator<char*>);

  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(a)), std::reverse_iterator<int* const*>);
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(b)), std::reverse_iterator<const int(*)[2]>);
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(c)), const long*);
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(d)), const long*);
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(e)), std::basic_const_iterator<int*>);

  return true;
}

constexpr bool testArray() {
  int a[2];
  int b[2][2];
  NonborrowingRange c[2];

  assert(std::ranges::cbegin(a) == a);
  assert(std::ranges::cbegin(b) == b);
  assert(std::ranges::cbegin(c) == c);

  assert(std::ranges::crbegin(a).base() == a + 2);
  assert(std::ranges::crbegin(b).base() == b + 2);
  assert(std::ranges::crbegin(c).base() == c + 2);

  return true;
}

struct BorrowingRange {
  int* begin() const;
  int* end() const;
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowingRange> = true;

static_assert(std::is_invocable_v<RangeCBeginT, BorrowingRange>);
static_assert(std::is_invocable_v<RangeCBeginT, const BorrowingRange>);
static_assert(std::is_invocable_v<RangeCBeginT, BorrowingRange&>);
static_assert(std::is_invocable_v<RangeCBeginT, const BorrowingRange&>);
static_assert(std::is_invocable_v<RangeCRBeginT, BorrowingRange>);
static_assert(std::is_invocable_v<RangeCRBeginT, const BorrowingRange>);
static_assert(std::is_invocable_v<RangeCRBeginT, BorrowingRange&>);
static_assert(std::is_invocable_v<RangeCRBeginT, const BorrowingRange&>);

struct NoThrowBeginThrowingEnd {
  const int* begin() const noexcept;
  const int* end() const;
} ntbte;
static_assert(noexcept(std::ranges::cbegin(ntbte)));
static_assert(!noexcept(std::ranges::crbegin(ntbte)));

struct ThrowingBeginNoThrowEnd {
  const int* begin() const;
  const int* end() const noexcept;
} tbnte;
static_assert(!noexcept(std::ranges::cbegin(tbnte)));
static_assert(noexcept(std::ranges::crbegin(tbnte)));

// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder {
  T t;
};
static_assert(!std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*&>);

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
  static_assert(testArray());

  return 0;
}
