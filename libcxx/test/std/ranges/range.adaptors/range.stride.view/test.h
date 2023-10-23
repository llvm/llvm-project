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
#include "__iterator/default_sentinel.h"
#include "__ranges/access.h"
#include "__ranges/concepts.h"
#include "__ranges/enable_borrowed_range.h"
#include "__ranges/enable_view.h"
#include "__ranges/size.h"
#include "__ranges/stride_view.h"
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

  friend constexpr bool operator==(const Derived&, const Derived&) { return true; }
  friend constexpr bool operator==(const std::default_sentinel_t&, const Derived&) { return true; }
  friend constexpr bool operator==(const Derived&, const std::default_sentinel_t&) { return true; }
};

struct NotSimpleViewIter : InputIterBase<NotSimpleViewIter> {
  constexpr NotSimpleViewIter()                                    = default;
  constexpr NotSimpleViewIter(const NotSimpleViewIter&)            = default;
  constexpr NotSimpleViewIter(NotSimpleViewIter&&)                 = default;
  constexpr NotSimpleViewIter& operator=(NotSimpleViewIter&&)      = default;
  constexpr NotSimpleViewIter& operator=(const NotSimpleViewIter&) = default;
};

struct NotSimpleViewIterEnd : InputIterBase<NotSimpleViewIterEnd> {
  constexpr NotSimpleViewIterEnd()                                       = default;
  constexpr NotSimpleViewIterEnd(const NotSimpleViewIterEnd&)            = default;
  constexpr NotSimpleViewIterEnd(NotSimpleViewIterEnd&&)                 = default;
  constexpr NotSimpleViewIterEnd& operator=(NotSimpleViewIterEnd&&)      = default;
  constexpr NotSimpleViewIterEnd& operator=(const NotSimpleViewIterEnd&) = default;
};

template <bool Convertible>
struct NotSimpleViewConstIter : InputIterBase<NotSimpleViewConstIter<Convertible>> {
  constexpr NotSimpleViewConstIter()                              = default;
  constexpr NotSimpleViewConstIter(const NotSimpleViewConstIter&) = default;
  constexpr NotSimpleViewConstIter(const NotSimpleViewIter&)
    requires Convertible
  {}
  constexpr NotSimpleViewConstIter(NotSimpleViewConstIter&&) = default;

  constexpr NotSimpleViewConstIter(NotSimpleViewIterEnd&&) = delete;

  constexpr NotSimpleViewConstIter& operator=(NotSimpleViewConstIter&&)      = default;
  constexpr NotSimpleViewConstIter& operator=(const NotSimpleViewConstIter&) = default;
};

template <bool Convertible>
constexpr bool operator==(const NotSimpleViewConstIter<Convertible>&, const NotSimpleViewIterEnd&) {
  return true;
}
template <bool Convertible>
constexpr bool operator==(const NotSimpleViewIterEnd&, const NotSimpleViewConstIter<Convertible>&) {
  return true;
}

constexpr bool operator==(const NotSimpleViewIter&, const NotSimpleViewIterEnd&) { return true; }
constexpr bool operator==(const NotSimpleViewIterEnd&, const NotSimpleViewIter&) { return true; }

template <bool Convertible = false>
struct NotSimpleViewDifferentBegin : std::ranges::view_base {
  constexpr NotSimpleViewConstIter<Convertible> begin() const { return {}; }
  constexpr NotSimpleViewIter begin() { return {}; }
  constexpr NotSimpleViewIterEnd end() const { return {}; }
  constexpr NotSimpleViewIterEnd end() { return {}; }
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<NotSimpleViewDifferentBegin<true>> = true;
template <>
inline constexpr bool std::ranges::enable_borrowed_range<NotSimpleViewDifferentBegin<false>> = true;

/*
 * XXXArrayView classes for use throughout the stride view tests.
 */

template <typename T>
struct RandomAccessArrayView : std::ranges::view_base {
  T* begin_;
  T* end_;

  constexpr RandomAccessArrayView(T* b, T* e) : begin_(b), end_(e) {}

  constexpr random_access_iterator<T*> begin() { return random_access_iterator<T*>{begin_}; }
  constexpr random_access_iterator<const T*> begin() const { return random_access_iterator<const T*>{begin_}; }
  constexpr sentinel_wrapper<random_access_iterator<T*>> end() {
    return sentinel_wrapper<random_access_iterator<T*>>{random_access_iterator<T*>{end_}};
  }
  constexpr sentinel_wrapper<random_access_iterator<const T*>> end() const {
    return sentinel_wrapper<random_access_iterator<const T*>>{random_access_iterator<const T*>{end_}};
  }
  constexpr std::size_t size() const { return end_ - begin_; }
};
static_assert(std::ranges::view<RandomAccessArrayView<int>>);
static_assert(std::ranges::random_access_range<RandomAccessArrayView<int>>);
static_assert(std::copyable<RandomAccessArrayView<int>>);

template <typename T>
struct BidirArrayView : std::ranges::view_base {
  T* begin_;
  T* end_;

  constexpr BidirArrayView(T* b, T* e) : begin_(b), end_(e) {}

  constexpr bidirectional_iterator<T*> begin() { return bidirectional_iterator<T*>{begin_}; }
  constexpr bidirectional_iterator<const T*> begin() const { return bidirectional_iterator<const T*>{begin_}; }
  constexpr sentinel_wrapper<bidirectional_iterator<T*>> end() {
    return sentinel_wrapper<bidirectional_iterator<T*>>{bidirectional_iterator<T*>{end_}};
  }
  constexpr sentinel_wrapper<bidirectional_iterator<const T*>> end() const {
    return sentinel_wrapper<bidirectional_iterator<const T*>>{bidirectional_iterator<const T*>{end_}};
  }
};
static_assert(std::ranges::view<BidirArrayView<int>>);
static_assert(std::ranges::bidirectional_range<BidirArrayView<int>>);
static_assert(std::copyable<BidirArrayView<int>>);

template <typename T>
struct ForwardArrayView : public std::ranges::view_base {
  T* begin_;
  T* end_;

  constexpr ForwardArrayView(T* b, T* e) : begin_(b), end_(e) {}

  constexpr forward_iterator<T*> begin() { return forward_iterator<T*>{begin_}; }
  constexpr forward_iterator<const T*> begin() const { return forward_iterator<const T*>{begin_}; }
  constexpr sentinel_wrapper<forward_iterator<T*>> end() {
    return sentinel_wrapper<forward_iterator<T*>>{forward_iterator<T*>{end_}};
  }
  constexpr sentinel_wrapper<forward_iterator<const T*>> end() const {
    return sentinel_wrapper<forward_iterator<const T*>>{forward_iterator<const T*>{end_}};
  }
};
static_assert(std::ranges::view<ForwardArrayView<int>>);
static_assert(std::ranges::forward_range<ForwardArrayView<int>>);
static_assert(std::copyable<ForwardArrayView<int>>);

template <typename T>
struct InputArrayView : std::ranges::view_base {
  T* begin_;
  T* end_;

  constexpr InputArrayView(T* b, T* e) : begin_(b), end_(e) {}

  constexpr cpp20_input_iterator<T*> begin() { return cpp20_input_iterator<T*>{begin_}; }
  constexpr random_access_iterator<const T*> begin() const { return random_access_iterator<const T*>{begin_}; }
  constexpr sentinel_wrapper<cpp20_input_iterator<T*>> end() {
    return sentinel_wrapper<cpp20_input_iterator<T*>>{cpp20_input_iterator<T*>{end_}};
  }
  constexpr std::size_t size() const { return end_ - begin_; }
};
static_assert(std::ranges::view<InputArrayView<int>>);
static_assert(std::ranges::input_range<InputArrayView<int>>);
static_assert(std::copyable<InputArrayView<int>>);

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

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
