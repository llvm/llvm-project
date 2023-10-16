//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H

#include "__concepts/movable.h"
#include "__iterator/default_sentinel.h"
#include "__ranges/concepts.h"
#include "__ranges/enable_view.h"
#include "__ranges/size.h"
#include "test_iterators.h"
#include <iterator>
#include <ranges>

template <typename I, std::ranges::range_difference_t<I> D>
concept CanStrideView = requires {
  std::ranges::stride_view<I>{I{}, D};
};

template <typename T = int>
struct InstrumentedBasicRange {
  T* begin() const;
  T* end() const;
};

struct MovedCopiedTrackedView {
  constexpr explicit MovedCopiedTrackedView(bool* moved = nullptr, bool* copied = nullptr)
      : wasMoveInitialized_(moved), wasCopyInitialized_(copied) {}
  constexpr MovedCopiedTrackedView(MovedCopiedTrackedView const& other)
      : wasMoveInitialized_(other.wasMoveInitialized_), wasCopyInitialized_(other.wasCopyInitialized_) {
    *wasCopyInitialized_ = true;
  }
  constexpr MovedCopiedTrackedView(MovedCopiedTrackedView&& other)
      : wasMoveInitialized_(other.wasMoveInitialized_), wasCopyInitialized_(other.wasCopyInitialized_) {
    *wasMoveInitialized_ = true;
  }
  MovedCopiedTrackedView& operator=(MovedCopiedTrackedView const&) = default;
  MovedCopiedTrackedView& operator=(MovedCopiedTrackedView&&)      = default;

  bool* wasMoveInitialized_ = nullptr;
  bool* wasCopyInitialized_ = nullptr;
};

template <typename T = int>
struct MovedCopiedTrackedBasicView : MovedCopiedTrackedView, std::ranges::view_base {
  constexpr explicit MovedCopiedTrackedBasicView(T* b, T* e, bool* moved = nullptr, bool* copied = nullptr)
      : MovedCopiedTrackedView(moved, copied), begin_(b), end_(e) {}
  constexpr MovedCopiedTrackedBasicView(const MovedCopiedTrackedBasicView& other)
      : MovedCopiedTrackedView(other), begin_(other.begin_), end_(other.end_) {}
  constexpr MovedCopiedTrackedBasicView(MovedCopiedTrackedBasicView&& other)
      : MovedCopiedTrackedView(std::move(other)), begin_(other.begin_), end_(other.end_) {}
  MovedCopiedTrackedBasicView& operator=(MovedCopiedTrackedBasicView const&) = default;
  MovedCopiedTrackedBasicView& operator=(MovedCopiedTrackedBasicView&&)      = default;
  constexpr T* begin() const { return begin_; }
  constexpr T* end() const { return end_; }

  T* begin_;
  T* end_;
};

template <typename T>
MovedCopiedTrackedBasicView(T, T, bool*, bool*) -> MovedCopiedTrackedBasicView<T>;

template <typename T>
struct InstrumentedBorrowedRange : public MovedCopiedTrackedBasicView<T> {};

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<InstrumentedBorrowedRange<T>> = true;

template <typename T = int>
struct MovedOnlyTrackedBasicView : MovedCopiedTrackedView, std::ranges::view_base {
  constexpr explicit MovedOnlyTrackedBasicView(T* b, T* e, bool* moved = nullptr, bool* copied = nullptr)
      : MovedCopiedTrackedView(moved, copied), begin_(b), end_(e) {}
  constexpr MovedOnlyTrackedBasicView(MovedOnlyTrackedBasicView&& other)
      : MovedCopiedTrackedView(std::move(other)), begin_(other.begin_), end_(other.end_) {}
  MovedOnlyTrackedBasicView& operator=(MovedOnlyTrackedBasicView const&) = delete;
  MovedOnlyTrackedBasicView& operator=(MovedOnlyTrackedBasicView&&)      = default;
  constexpr T* begin() const { return begin_; }
  constexpr T* end() const { return end_; }

  T* begin_;
  T* end_;
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
  friend constexpr bool operator==(const std::default_sentinel_t&, const ForwardIterBase&) { return true; }
  friend constexpr bool operator==(const ForwardIterBase&, const std::default_sentinel_t&) { return true; }
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
  friend constexpr bool operator==(const std::default_sentinel_t&, const InputIterBase&) { return true; }
  friend constexpr bool operator==(const InputIterBase&, const std::default_sentinel_t&) { return true; }
};

struct NotSimpleViewIter : ForwardIterBase<NotSimpleViewIter> {
  constexpr NotSimpleViewIter()                                    = default;
  constexpr NotSimpleViewIter(const NotSimpleViewIter&)            = default;
  constexpr NotSimpleViewIter(NotSimpleViewIter&&)                 = default;
  constexpr NotSimpleViewIter& operator=(NotSimpleViewIter&&)      = default;
  constexpr NotSimpleViewIter& operator=(const NotSimpleViewIter&) = default;
};

struct NotSimpleViewIterEnd : ForwardIterBase<NotSimpleViewIter> {
  constexpr NotSimpleViewIterEnd()                                       = default;
  constexpr NotSimpleViewIterEnd(const NotSimpleViewIterEnd&)            = default;
  constexpr NotSimpleViewIterEnd(NotSimpleViewIterEnd&&)                 = default;
  constexpr NotSimpleViewIterEnd& operator=(NotSimpleViewIterEnd&&)      = default;
  constexpr NotSimpleViewIterEnd& operator=(const NotSimpleViewIterEnd&) = default;
};

struct ConstNotSimpleViewIter : ForwardIterBase<ConstNotSimpleViewIter> {
  constexpr ConstNotSimpleViewIter()                              = default;
  constexpr ConstNotSimpleViewIter(const ConstNotSimpleViewIter&) = default;
  constexpr ConstNotSimpleViewIter(const NotSimpleViewIter&) {}
  constexpr ConstNotSimpleViewIter(ConstNotSimpleViewIter&&) = default;

  constexpr ConstNotSimpleViewIter(NotSimpleViewIter&&) {}
  constexpr ConstNotSimpleViewIter(NotSimpleViewIterEnd&&) = delete;

  constexpr ConstNotSimpleViewIter& operator=(ConstNotSimpleViewIter&&)      = default;
  constexpr ConstNotSimpleViewIter& operator=(const ConstNotSimpleViewIter&) = default;
};

struct NotSimpleView : std::ranges::view_base {
  constexpr ConstNotSimpleViewIter begin() const { return {}; }
  constexpr NotSimpleViewIter begin() { return {}; }
  constexpr ConstNotSimpleViewIter end() const { return {}; }
  constexpr NotSimpleViewIterEnd end() { return {}; }
};

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

static_assert(std::ranges::view<BidirView>);
static_assert(std::ranges::bidirectional_range<BidirView>);
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
static_assert(std::ranges::forward_range<ForwardView>);
static_assert(std::copyable<ForwardView>);

struct InputView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr InputView(int* b, int* e) : begin_(b), end_(e) {}

  constexpr cpp20_input_iterator<int*> begin() { return cpp20_input_iterator<int*>{begin_}; }
  //constexpr random_access_iterator<const int*> begin() const { return random_access_iterator<const int*>{begin_}; }
  constexpr sentinel_wrapper<cpp20_input_iterator<int*>> end() {
    return sentinel_wrapper<cpp20_input_iterator<int*>>{cpp20_input_iterator<int*>{end_}};
  }
  constexpr std::size_t size() const { return end_ - begin_; }
};

static_assert(std::ranges::view<InputView>);
static_assert(std::ranges::input_range<InputView>);
static_assert(std::copyable<InputView>);

struct UnsizedBasicRangeIterator : ForwardIterBase<UnsizedBasicRangeIterator> {};

struct UnsizedBasicRange : std::ranges::view_base {
  UnsizedBasicRangeIterator begin() const;
  UnsizedBasicRangeIterator end() const;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
