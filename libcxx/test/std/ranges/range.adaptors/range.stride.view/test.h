//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H

#include "test_iterators.h"
#include <iterator>
#include <ranges>

template <typename T = int>
struct Range : std::ranges::view_base {
  constexpr explicit Range(T* b, T* e) : begin_(b), end_(e) {}
  constexpr Range(Range const& other) : begin_(other.begin_), end_(other.end_), wasCopyInitialized(true) {}
  constexpr Range(Range&& other) : begin_(other.begin_), end_(other.end_), wasMoveInitialized(true) {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;
  constexpr T* begin() const { return begin_; }
  constexpr T* end() const { return end_; }

  T* begin_;
  T* end_;
  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

template <typename T>
Range(T, T) -> Range<T>;

template <typename T>
struct BorrowedRange : public Range<T> {};

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange<T>> = true;

struct NoCopyRange : std::ranges::view_base {
  explicit NoCopyRange(int*, int*);
  NoCopyRange(NoCopyRange const&)            = delete;
  NoCopyRange(NoCopyRange&&)                 = default;
  NoCopyRange& operator=(NoCopyRange const&) = default;
  NoCopyRange& operator=(NoCopyRange&&)      = default;
  int* begin() const;
  int* end() const;
};

template <class Derived>
struct ForwardIterBase {
  using iterator_concept = std::forward_iterator_tag;
  using value_type       = int;
  using difference_type  = std::intptr_t;

  constexpr int operator*() const { return 5; }

  constexpr Derived& operator++() { return static_cast<Derived&>(*this); }
  constexpr Derived operator++(int) { return {}; }

  friend constexpr bool operator==(const ForwardIterBase&, const ForwardIterBase&) { return true; }
};

template <class Derived>
struct InputIterBase {
  using iterator_concept = std::input_iterator_tag;
  using value_type       = int;
  using difference_type  = std::intptr_t;

  constexpr int operator*() const { return 5; }

  constexpr Derived& operator++() { return static_cast<Derived&>(*this); }
  constexpr Derived operator++(int) { return {}; }

  friend constexpr bool operator==(const InputIterBase&, const InputIterBase&) { return true; }
};

struct NotSimpleViewIterB : ForwardIterBase<NotSimpleViewIterB> {
  bool moved = false;

  constexpr NotSimpleViewIterB()                          = default;
  constexpr NotSimpleViewIterB(const NotSimpleViewIterB&) = default;
  constexpr NotSimpleViewIterB(NotSimpleViewIterB&&) : moved{true} {}
  constexpr NotSimpleViewIterB& operator=(NotSimpleViewIterB&&)      = default;
  constexpr NotSimpleViewIterB& operator=(const NotSimpleViewIterB&) = default;
};

struct NotSimpleViewIterA : ForwardIterBase<NotSimpleViewIterA> {
  bool moved         = false;
  bool moved_from_a  = false;
  bool copied_from_a = false;

  constexpr NotSimpleViewIterA()                          = default;
  constexpr NotSimpleViewIterA(const NotSimpleViewIterA&) = default;
  constexpr NotSimpleViewIterA(const NotSimpleViewIterB&) : copied_from_a{true} {}
  constexpr NotSimpleViewIterA(NotSimpleViewIterA&&) : moved{true} {}
  constexpr NotSimpleViewIterA(NotSimpleViewIterB&&) : moved_from_a{true} {}
  constexpr NotSimpleViewIterA& operator=(NotSimpleViewIterA&&)      = default;
  constexpr NotSimpleViewIterA& operator=(const NotSimpleViewIterA&) = default;
};

struct NotSimpleView : std::ranges::view_base {
  constexpr NotSimpleViewIterA begin() const { return {}; }
  constexpr NotSimpleViewIterB begin() { return {}; }
  constexpr NotSimpleViewIterA end() const { return {}; }
  constexpr NotSimpleViewIterB end() { return {}; }

  int* begin_;
  int* end_;
  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

struct ForwardTracedMoveIter : ForwardIterBase<ForwardTracedMoveIter> {
  bool moved = false;

  constexpr ForwardTracedMoveIter()                             = default;
  constexpr ForwardTracedMoveIter(const ForwardTracedMoveIter&) = default;
  constexpr ForwardTracedMoveIter(ForwardTracedMoveIter&&) : moved{true} {}
  constexpr ForwardTracedMoveIter& operator=(ForwardTracedMoveIter&&)      = default;
  constexpr ForwardTracedMoveIter& operator=(const ForwardTracedMoveIter&) = default;
};

struct ForwardTracedMoveView : std::ranges::view_base {
  constexpr ForwardTracedMoveIter begin() const { return {}; }
  constexpr ForwardTracedMoveIter end() const { return {}; }
};

struct BidirRange : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr BidirRange(int* b, int* e) : begin_(b), end_(e) {}

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{begin_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{begin_}; }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>{end_}; }
  constexpr bidirectional_iterator<const int*> end() const { return bidirectional_iterator<const int*>{end_}; }
};
static_assert(std::ranges::bidirectional_range<BidirRange>);
static_assert(std::ranges::common_range<BidirRange>);
static_assert(std::ranges::view<BidirRange>);
static_assert(std::copyable<BidirRange>);

enum CopyCategory { MoveOnly, Copyable };
template <CopyCategory CC>
struct BidirSentRange : std::ranges::view_base {
  using sent_t       = sentinel_wrapper<bidirectional_iterator<int*>>;
  using sent_const_t = sentinel_wrapper<bidirectional_iterator<const int*>>;

  int* begin_;
  int* end_;

  constexpr BidirSentRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr BidirSentRange(const BidirSentRange&)
    requires(CC == Copyable)
  = default;
  constexpr BidirSentRange(BidirSentRange&&)
    requires(CC == MoveOnly)
  = default;
  constexpr BidirSentRange& operator=(const BidirSentRange&)
    requires(CC == Copyable)
  = default;
  constexpr BidirSentRange& operator=(BidirSentRange&&)
    requires(CC == MoveOnly)
  = default;

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{begin_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{begin_}; }
  constexpr sent_t end() { return sent_t{bidirectional_iterator<int*>{end_}}; }
  constexpr sent_const_t end() const { return sent_const_t{bidirectional_iterator<const int*>{end_}}; }
};
static_assert(std::ranges::bidirectional_range<BidirSentRange<MoveOnly>>);
static_assert(!std::ranges::common_range<BidirSentRange<MoveOnly>>);
static_assert(std::ranges::view<BidirSentRange<MoveOnly>>);
static_assert(!std::copyable<BidirSentRange<MoveOnly>>);
static_assert(std::ranges::bidirectional_range<BidirSentRange<Copyable>>);
static_assert(!std::ranges::common_range<BidirSentRange<Copyable>>);
static_assert(std::ranges::view<BidirSentRange<Copyable>>);
static_assert(std::copyable<BidirSentRange<Copyable>>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
