//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::cend
// std::ranges::crend

#include <ranges>

#include <cassert>
#include <utility>
#include "almost_satisfies_types.h"
#include "test_macros.h"
#include "test_iterators.h"

using RangeCEndT  = decltype(std::ranges::cend);
using RangeCREndT = decltype(std::ranges::crend);

static_assert(!std::is_invocable_v<RangeCEndT, int (&&)[10]>);
static_assert(std::is_invocable_v<RangeCEndT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCEndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCEndT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, int (&&)[10]>);
static_assert(std::is_invocable_v<RangeCREndT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCREndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, int (&)[]>);

static_assert(!std::is_invocable_v<RangeCEndT, InputRangeNotDerivedFrom>);
static_assert(!std::is_invocable_v<RangeCEndT, InputRangeNotIndirectlyReadable>);
static_assert(!std::is_invocable_v<RangeCEndT, InputRangeNotInputOrOutputIterator>);
static_assert(!std::is_invocable_v<RangeCEndT, InputRangeNotSentinelSemiregular>);
static_assert(!std::is_invocable_v<RangeCEndT, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!std::is_invocable_v<RangeCREndT, BidirectionalRangeNotDerivedFrom>);
static_assert(!std::is_invocable_v<RangeCREndT, BidirectionalRangeNotSentinelSemiregular>);
static_assert(!std::is_invocable_v<RangeCREndT, BidirectionalRangeNotSentinelWeaklyEqualityComparableWith>);
static_assert(!std::is_invocable_v<RangeCREndT, BidirectionalRangeNotDecrementable>);

struct Incomplete;

static_assert(!std::is_invocable_v<RangeCEndT, Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCEndT, const Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCEndT, Incomplete (&&)[10]>);
static_assert(!std::is_invocable_v<RangeCEndT, const Incomplete (&&)[10]>);

static_assert(!std::is_invocable_v<RangeCREndT, Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, const Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, Incomplete (&&)[10]>);
static_assert(!std::is_invocable_v<RangeCREndT, const Incomplete (&&)[10]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCEndT, Incomplete (&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCEndT, const Incomplete (&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCREndT, Incomplete (&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCREndT, const Incomplete (&)[]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCEndT, Incomplete (&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCEndT, const Incomplete (&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCREndT, Incomplete (&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCREndT, const Incomplete (&)[10]>);

struct NonborrowingRange {
  int x;
  constexpr const int* begin() const { return &x; }
  constexpr const int* rbegin() const { return &x; }
  constexpr const int* end() const { return &x; }
  constexpr const int* rend() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert(std::is_invocable_v<RangeCEndT, NonborrowingRange&>);
static_assert(!std::is_invocable_v<RangeCEndT, NonborrowingRange&&>);
static_assert(std::is_invocable_v<RangeCEndT, NonborrowingRange const&>);
static_assert(!std::is_invocable_v<RangeCEndT, NonborrowingRange const&&>);
static_assert(std::is_invocable_v<RangeCREndT, NonborrowingRange&>);
static_assert(!std::is_invocable_v<RangeCREndT, NonborrowingRange&&>);
static_assert(std::is_invocable_v<RangeCREndT, NonborrowingRange const&>);
static_assert(!std::is_invocable_v<RangeCREndT, NonborrowingRange const&&>);

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

  ASSERT_SAME_TYPE(decltype(std::ranges::cend(a)), int* const*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(b)), const int(*)[2]);
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(c)), const short*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(d)), const short*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(e)), std::basic_const_iterator<char*>);

  ASSERT_SAME_TYPE(decltype(std::ranges::crend(a)), std::reverse_iterator<int* const*>);
  ASSERT_SAME_TYPE(decltype(std::ranges::crend(b)), std::reverse_iterator<const int(*)[2]>);
  ASSERT_SAME_TYPE(decltype(std::ranges::crend(c)), const long*);
  ASSERT_SAME_TYPE(decltype(std::ranges::crend(d)), const long*);
  ASSERT_SAME_TYPE(decltype(std::ranges::crend(e)), std::basic_const_iterator<int*>);

  return true;
}

constexpr bool testArray() {
  int a[2];
  int b[2][2];
  NonborrowingRange c[2];

  assert(std::ranges::cend(a) == a + 2);
  assert(std::ranges::cend(b) == b + 2);
  assert(std::ranges::cend(c) == c + 2);

  assert(std::ranges::crend(a).base() == a);
  assert(std::ranges::crend(b).base() == b);
  assert(std::ranges::crend(c).base() == c);

  return true;
}

struct BorrowingRange {
  int* begin() const;
  int* end() const;
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowingRange> = true;

static_assert(std::is_invocable_v<RangeCEndT, BorrowingRange>);
static_assert(std::is_invocable_v<RangeCEndT, const BorrowingRange>);
static_assert(std::is_invocable_v<RangeCEndT, BorrowingRange&>);
static_assert(std::is_invocable_v<RangeCEndT, const BorrowingRange&>);
static_assert(std::is_invocable_v<RangeCREndT, BorrowingRange>);
static_assert(std::is_invocable_v<RangeCREndT, const BorrowingRange>);
static_assert(std::is_invocable_v<RangeCREndT, BorrowingRange&>);
static_assert(std::is_invocable_v<RangeCREndT, const BorrowingRange&>);

struct NoThrowEndThrowingEnd {
  const int* begin() const noexcept;
  const int* end() const;
} ntbte;
static_assert(!noexcept(std::ranges::cend(ntbte)));
static_assert(noexcept(std::ranges::crend(ntbte)));

struct ThrowingEndNoThrowEnd {
  const int* begin() const;
  const int* end() const noexcept;
} tbnte;
static_assert(noexcept(std::ranges::cend(tbnte)));
static_assert(!noexcept(std::ranges::crend(tbnte)));

// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder {
  T t;
};
static_assert(!std::is_invocable_v<RangeCEndT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCEndT, Holder<Incomplete>*&>);
static_assert(!std::is_invocable_v<RangeCREndT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCREndT, Holder<Incomplete>*&>);

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
  static_assert(testArray());

  return 0;
}
