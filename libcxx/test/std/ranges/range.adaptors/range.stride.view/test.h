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
struct InstrumentedBasicRange {
  T* begin() const;
  T* end() const;
};

template <typename T = int>
struct InstrumentedBasicView : std::ranges::view_base {
  constexpr explicit InstrumentedBasicView(T* b, T* e) : begin_(b), end_(e) {}
  constexpr InstrumentedBasicView(InstrumentedBasicView const& other)
      : begin_(other.begin_), end_(other.end_), wasCopyInitialized(true) {}
  constexpr InstrumentedBasicView(InstrumentedBasicView&& other)
      : begin_(other.begin_), end_(other.end_), wasMoveInitialized(true) {}
  InstrumentedBasicView& operator=(InstrumentedBasicView const&) = default;
  InstrumentedBasicView& operator=(InstrumentedBasicView&&)      = default;
  constexpr T* begin() const { return begin_; }
  constexpr T* end() const { return end_; }

  T* begin_;
  T* end_;
  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

template <typename T>
InstrumentedBasicView(T, T) -> InstrumentedBasicView<T>;

template <typename T>
struct InstrumentedBorrowedRange : public InstrumentedBasicView<T> {};

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<InstrumentedBorrowedRange<T>> = true;

struct NoCopyView : std::ranges::view_base {
  explicit NoCopyView(int*, int*);
  NoCopyView(NoCopyView const&)            = delete;
  NoCopyView(NoCopyView&&)                 = default;
  NoCopyView& operator=(NoCopyView const&) = default;
  NoCopyView& operator=(NoCopyView&&)      = default;
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

struct InstrumentedNotSimpleView : std::ranges::view_base {
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

struct BidirView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr BidirView(int* b, int* e) : begin_(b), end_(e) {}

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{begin_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{begin_}; }
  constexpr sentinel_wrapper<bidirectional_iterator<int*>> end() {
    return sentinel_wrapper<bidirectional_iterator<int*>>{bidirectional_iterator<int*>{end_}};
  }
  constexpr sentinel_wrapper<bidirectional_iterator<const int*>> end() const {
    return sentinel_wrapper<bidirectional_iterator<const int*>>{bidirectional_iterator<const int*>{end_}};
  }
};

static_assert(std::ranges::view<BidirView>);
static_assert(std::copyable<BidirView>);

struct ForwardView : public std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr ForwardView(int* b, int* e) : begin_(b), end_(e) {}

  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>{begin_}; }
  constexpr forward_iterator<const int*> begin() const { return forward_iterator<const int*>{begin_}; }
  constexpr sentinel_wrapper<forward_iterator<int*>> end() {
    return sentinel_wrapper<forward_iterator<int*>>{forward_iterator<int*>{end_}};
  }
  constexpr sentinel_wrapper<forward_iterator<const int*>> end() const {
    return sentinel_wrapper<forward_iterator<const int*>>{forward_iterator<const int*>{end_}};
  }
};

static_assert(std::ranges::view<ForwardView>);
static_assert(std::copyable<ForwardView>);

struct RandomAccessView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr RandomAccessView(int* b, int* e) : begin_(b), end_(e) {}

  constexpr random_access_iterator<int*> begin() { return random_access_iterator<int*>{begin_}; }
  //constexpr random_access_iterator<const int*> begin() const { return random_access_iterator<const int*>{begin_}; }
  constexpr sentinel_wrapper<random_access_iterator<int*>> end() {
    return sentinel_wrapper<random_access_iterator<int*>>{random_access_iterator<int*>{end_}};
  }
  //constexpr sentinel_wrapper<random_access_iterator<const int*>> end() const { return sentinel_wrapper<random_access_iterator<const int*>>{random_access_iterator<const int*>{end_}}; }
  constexpr std::size_t size() const { return end_ - begin_; }
};

static_assert(std::ranges::view<RandomAccessView>);
static_assert(std::ranges::random_access_range<RandomAccessView>);
static_assert(std::copyable<RandomAccessView>);

/*
enum CopyCategory { MoveOnly, Copyable };
template <CopyCategory CC>
struct BidirSentView : std::ranges::view_base {
  using sent_t       = sentinel_wrapper<bidirectional_iterator<int*>>;
  using sent_const_t = sentinel_wrapper<bidirectional_iterator<const int*>>;

  int* begin_;
  int* end_;

  constexpr BidirSentView(int* b, int* e) : begin_(b), end_(e) {}
  constexpr BidirSentView(const BidirSentView&)
    requires(CC == Copyable)
  = default;
  constexpr BidirSentView(BidirSentView&&)
    requires(CC == MoveOnly)
  = default;
  constexpr BidirSentView& operator=(const BidirSentView&)
    requires(CC == Copyable)
  = default;
  constexpr BidirSentView& operator=(BidirSentView&&)
    requires(CC == MoveOnly)
  = default;

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{begin_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{begin_}; }
  constexpr sent_t end() { return sent_t{bidirectional_iterator<int*>{end_}}; }
  constexpr sent_const_t end() const { return sent_const_t{bidirectional_iterator<const int*>{end_}}; }
};
// TODO: Clean up.
static_assert(std::ranges::bidirectional_range<BidirSentView<MoveOnly>>);
static_assert(!std::ranges::common_range<BidirSentView<MoveOnly>>);
static_assert(std::ranges::view<BidirSentView<MoveOnly>>);
static_assert(!std::copyable<BidirSentView<MoveOnly>>);
static_assert(std::ranges::bidirectional_range<BidirSentView<Copyable>>);
static_assert(!std::ranges::common_range<BidirSentView<Copyable>>);
static_assert(std::ranges::view<BidirSentView<Copyable>>);
static_assert(std::copyable<BidirSentView<Copyable>>);
*/

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
