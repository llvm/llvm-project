//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H

#include "__concepts/equality_comparable.h"
#include "__concepts/movable.h"
#include "__concepts/semiregular.h"
#include "__iterator/concepts.h"
#include "__iterator/default_sentinel.h"
#include "__ranges/access.h"
#include "__ranges/concepts.h"
#include "__ranges/enable_borrowed_range.h"
#include "__ranges/enable_view.h"
#include "__ranges/size.h"
#include "__ranges/stride_view.h"
#include "test_iterators.h"
#include "test_range.h"
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

class non_view_range {
public:
  constexpr int* begin() const { return nullptr; }
  constexpr int* end() const { return nullptr; }
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

  friend constexpr bool operator==(const Derived&, const Derived&) { return true; }
};

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>>
struct MoveOnlyView : std::ranges::view_base {
  T begin_;
  T end_;

  constexpr explicit MoveOnlyView(T b, T e) : begin_(b), end_(e) {}

  constexpr MoveOnlyView(const MoveOnlyView&)            = delete;
  constexpr MoveOnlyView(MoveOnlyView&& other)           = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  constexpr MoveOnlyView& operator=(const MoveOnlyView&) = delete;

  constexpr T begin() const { return begin_; }
  constexpr sentinel_wrapper<T> end() const { return sentinel_wrapper<T>{end_}; }
};
static_assert(std::ranges::view<MoveOnlyView<cpp17_input_iterator<int*>>>);
static_assert(!std::copyable<MoveOnlyView<cpp17_input_iterator<int*>>>);

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>>
struct CopyableView : std::ranges::view_base {
  T begin_;
  T end_;

  constexpr explicit CopyableView(T b, T e) : begin_(b), end_(e) {}

  constexpr CopyableView(const CopyableView&)            = default;
  constexpr CopyableView& operator=(const CopyableView&) = default;

  constexpr T begin() const { return begin_; }
  constexpr sentinel_wrapper<T> end() const { return sentinel_wrapper<T>{end_}; }
};
static_assert(std::ranges::view<CopyableView<cpp17_input_iterator<int*>>>);
static_assert(std::copyable<CopyableView<cpp17_input_iterator<int*>>>);

//TODO: Rename as View.
template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>>
struct InputView : std::ranges::view_base {
  T begin_;
  T end_;

  constexpr InputView(T b, T e) : begin_(b), end_(e) {}

  constexpr T begin() { return begin_; }
  constexpr T begin() const { return begin_; }
  constexpr sentinel_wrapper<T> end() { return sentinel_wrapper<T>{end_}; }
  constexpr sentinel_wrapper<T> end() const { return sentinel_wrapper<T>{end_}; }
};

// Don't move/hold the iterator itself, move/hold the base
// of that iterator and reconstruct the iterator on demand.
// May result in aliasing (if, e.g., Iterator is an iterator
// over int *).
template <class Iterator, class Sentinel>
struct ViewOverNonCopyable : std::ranges::view_base {
  constexpr explicit ViewOverNonCopyable(Iterator it, Sentinel sent)
      : it_(base(std::move(it))), sent_(base(std::move(sent))) {}

  ViewOverNonCopyable(ViewOverNonCopyable&&)            = default;
  ViewOverNonCopyable& operator=(ViewOverNonCopyable&&) = default;

  constexpr Iterator begin() const { return Iterator(it_); }
  constexpr Sentinel end() const { return Sentinel(sent_); }

private:
  decltype(base(std::declval<Iterator>())) it_;
  decltype(base(std::declval<Sentinel>())) sent_;
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

struct UnsizedBasicRangeIterator : ForwardIterBase<UnsizedBasicRangeIterator> {};

struct UnsizedBasicRange : std::ranges::view_base {
  UnsizedBasicRangeIterator begin() const;
  UnsizedBasicRangeIterator end() const;
};

// TODO: Cleanup
struct SizedInputIterator {
  using iterator_concept = std::input_iterator_tag;
  using value_type       = int;
  using difference_type  = std::intptr_t;

  int* __v_;

  constexpr SizedInputIterator() { __v_ = nullptr; }
  constexpr SizedInputIterator(int* v) { __v_ = v; }
  constexpr SizedInputIterator(const SizedInputIterator& sii) { __v_ = sii.__v_; }

  constexpr int operator*() const { return *__v_; }
  constexpr SizedInputIterator& operator++() {
    __v_++;
    return *this;
  }
  constexpr SizedInputIterator operator++(int) {
    auto nv = __v_;
    nv++;
    return SizedInputIterator(nv);
  }
  friend constexpr bool operator==(const SizedInputIterator& left, const SizedInputIterator& right) {
    return left.__v_ == right.__v_;
  }
  friend constexpr difference_type operator-(const SizedInputIterator& left, const SizedInputIterator& right) {
    return left.__v_ - right.__v_;
  }
};
static_assert(std::input_iterator<SizedInputIterator>);
static_assert(std::sized_sentinel_for<SizedInputIterator, SizedInputIterator>);

// TODO: Cleanup
struct SizedForwardIterator {
  using iterator_concept = std::forward_iterator_tag;
  using value_type       = int;
  using difference_type  = std::intptr_t;

  int* __v_;

  constexpr SizedForwardIterator() { __v_ = nullptr; }
  constexpr SizedForwardIterator(int* v) { __v_ = v; }
  constexpr SizedForwardIterator(const SizedInputIterator& sii) { __v_ = sii.__v_; }

  constexpr int operator*() const { return *__v_; }
  constexpr SizedForwardIterator& operator++() {
    __v_++;
    return *this;
  }
  constexpr SizedForwardIterator operator++(int) {
    auto nv = __v_;
    nv++;
    return SizedForwardIterator(nv);
  }
  friend constexpr bool operator==(const SizedForwardIterator& left, const SizedForwardIterator& right) {
    return left.__v_ == right.__v_;
  }
  friend constexpr difference_type operator-(const SizedForwardIterator& left, const SizedForwardIterator& right) {
    return left.__v_ - right.__v_;
  }
};
static_assert(std::input_iterator<SizedForwardIterator>);
static_assert(std::sized_sentinel_for<SizedForwardIterator, SizedForwardIterator>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
