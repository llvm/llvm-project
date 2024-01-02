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
#include "test_macros.h"
#include "test_range.h"
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

// Iterators

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

// Don't move/hold the iterator itself, copy/hold the base
// of that iterator and reconstruct the iterator on demand.
// May result in aliasing (if, e.g., Iterator is an iterator
// over int *).
template <class Iterator, class Sentinel>
struct ViewOverNonCopyableIterator : std::ranges::view_base {
  constexpr explicit ViewOverNonCopyableIterator(Iterator it, Sentinel sent) : it_(base(it)), sent_(base(sent)) {}

  ViewOverNonCopyableIterator(ViewOverNonCopyableIterator&&)            = default;
  ViewOverNonCopyableIterator& operator=(ViewOverNonCopyableIterator&&) = default;

  constexpr Iterator begin() const { return Iterator(it_); }
  constexpr Sentinel end() const { return Sentinel(sent_); }

private:
  decltype(base(std::declval<Iterator>())) it_;
  decltype(base(std::declval<Sentinel>())) sent_;
};

struct UnsizedBasicRangeIterator : ForwardIterBase<UnsizedBasicRangeIterator> {};

struct SizedInputIterator {
  using iterator_concept = std::input_iterator_tag;
  using value_type       = int;
  using difference_type  = std::intptr_t;

  int* __v_ = nullptr;

  constexpr SizedInputIterator() = default;
  constexpr SizedInputIterator(int* v) { __v_ = v; }
  constexpr SizedInputIterator(const SizedInputIterator& sii)        = default;
  constexpr SizedInputIterator& operator=(const SizedInputIterator&) = default;
  constexpr SizedInputIterator& operator=(SizedInputIterator&&)      = default;

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

struct SizedForwardIterator {
  using iterator_concept = std::forward_iterator_tag;
  using value_type       = int;
  using difference_type  = std::intptr_t;

  int* __v_ = nullptr;

  constexpr SizedForwardIterator() = default;
  constexpr SizedForwardIterator(int* v) { __v_ = v; }
  constexpr SizedForwardIterator(const SizedForwardIterator& sii)        = default;
  constexpr SizedForwardIterator& operator=(const SizedForwardIterator&) = default;
  constexpr SizedForwardIterator& operator=(SizedForwardIterator&&)      = default;

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

template <bool Swappable = true, bool NoExcept = true>
struct MaybeExceptIterSwapIterator : InputIterBase<MaybeExceptIterSwapIterator<Swappable, NoExcept>> {
  int* counter_{nullptr};
  constexpr MaybeExceptIterSwapIterator()                                                    = default;
  constexpr MaybeExceptIterSwapIterator(const MaybeExceptIterSwapIterator&)                  = default;
  constexpr MaybeExceptIterSwapIterator(MaybeExceptIterSwapIterator&&)                       = default;
  constexpr MaybeExceptIterSwapIterator& operator=(const MaybeExceptIterSwapIterator& other) = default;
  constexpr MaybeExceptIterSwapIterator& operator=(MaybeExceptIterSwapIterator&& other)      = default;

  constexpr explicit MaybeExceptIterSwapIterator(int* counter) : counter_(counter) {}

  friend constexpr void iter_swap(const MaybeExceptIterSwapIterator& t, const MaybeExceptIterSwapIterator& u) noexcept
    requires Swappable && NoExcept
  {
    (*t.counter_)++;
    (*u.counter_)++;
  }

  friend constexpr void iter_swap(const MaybeExceptIterSwapIterator& t, const MaybeExceptIterSwapIterator& u)
    requires Swappable && (!NoExcept)
  {
    (*t.counter_)++;
    (*u.counter_)++;
  }

  friend constexpr int iter_move(const MaybeExceptIterSwapIterator& t) {
    (*t.counter_)++;
    return 5;
  }
  friend constexpr int iter_move(const MaybeExceptIterSwapIterator& t) noexcept
    requires NoExcept
  {
    (*t.counter_)++;
    return 5;
  }

  constexpr int operator*() const { return 5; }
};

template <bool Swappable = true, bool NoExcept = true>
struct IterSwapRange : std::ranges::view_base {
  MaybeExceptIterSwapIterator<Swappable, NoExcept> begin_;
  MaybeExceptIterSwapIterator<Swappable, NoExcept> end_;
  constexpr IterSwapRange(int* counter)
      : begin_(MaybeExceptIterSwapIterator<Swappable, NoExcept>(counter)),
        end_(MaybeExceptIterSwapIterator<Swappable, NoExcept>(counter)) {}
  constexpr MaybeExceptIterSwapIterator<Swappable, NoExcept> begin() const { return begin_; }
  constexpr MaybeExceptIterSwapIterator<Swappable, NoExcept> end() const { return end_; }
};

// Concepts

template <typename I, std::ranges::range_difference_t<I> D>
concept CanStrideView = requires {
  std::ranges::stride_view<I>{I{}, D};
};

// Views

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

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>>
struct BasicTestView : std::ranges::view_base {
  T begin_;
  T end_;

  constexpr BasicTestView(T b, T e) : begin_(b), end_(e) {}

  constexpr T begin() { return begin_; }
  constexpr T begin() const { return begin_; }
  constexpr S end() { return S{end_}; }
  constexpr S end() const { return S{end_}; }
};

struct UnsizedBasicView : std::ranges::view_base {
  UnsizedBasicRangeIterator begin() const;
  UnsizedBasicRangeIterator end() const;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
