//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H

#include <iterator>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "test_range.h"

template <class Derived, typename Iter = int*, bool Sized = false>
struct InputIterBase {
  using iterator_concept = std::input_iterator_tag;
  using value_type       = typename std::iterator_traits<Iter>::value_type;
  using difference_type  = typename std::iterator_traits<Iter>::difference_type;

  Iter value_{};

  constexpr InputIterBase()                                      = default;
  constexpr InputIterBase(const InputIterBase&)                  = default;
  constexpr InputIterBase(InputIterBase&&)                       = default;
  constexpr InputIterBase& operator=(const InputIterBase& other) = default;
  constexpr InputIterBase& operator=(InputIterBase&& other)      = default;
  constexpr explicit InputIterBase(Iter value) : value_(value) {}

  constexpr value_type operator*() const { return *value_; }
  constexpr Derived& operator++() {
    value_++;
    return static_cast<Derived&>(*this);
  }
  constexpr Derived operator++(int) {
    auto nv = *this;
    value_++;
    return nv;
  }
  friend constexpr bool operator==(const Derived& left, const Derived& right) { return left.value_ == right.value_; }
  friend constexpr difference_type operator-(const Derived& left, const Derived& right)
    requires Sized
  {
    return left.value_ - right.value_;
  }
};

struct UnsizedBasicRangeIterator : InputIterBase<UnsizedBasicRangeIterator> {};

struct SizedInputIterator : InputIterBase<SizedInputIterator, int*, true> {
  using InputIterBase::InputIterBase;
};
static_assert(std::input_iterator<SizedInputIterator>);
static_assert(std::sized_sentinel_for<SizedInputIterator, SizedInputIterator>);

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

// Put IterMoveIterSwapTestRangeIterator in a namespace to test ADL of CPOs iter_swap and iter_move
// (see iter_swap.pass.cpp and iter_move.pass.cpp).
namespace adl {
template <typename T = int*, bool IsSwappable = true, bool IsNoExcept = true>
struct IterMoveIterSwapTestRangeIterator
    : InputIterBase<IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept>, T, false> {
  int* counter_{nullptr};

  using InputIterBase<IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept>, T, false>::InputIterBase;

  constexpr IterMoveIterSwapTestRangeIterator(T value, int* counter)
      : InputIterBase<IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept>, T, false>(value),
        counter_(counter) {}

  friend constexpr void iter_swap(IterMoveIterSwapTestRangeIterator t, IterMoveIterSwapTestRangeIterator u) noexcept
    requires IsSwappable && IsNoExcept
  {
    (*t.counter_)++;
    (*u.counter_)++;
    std::swap(*t.value_, *u.value_);
  }

  friend constexpr void iter_swap(IterMoveIterSwapTestRangeIterator t, IterMoveIterSwapTestRangeIterator u)
    requires IsSwappable && (!IsNoExcept)
  {
    (*t.counter_)++;
    (*u.counter_)++;
    std::swap(*t.value_, *u.value_);
  }

  friend constexpr auto iter_move(const IterMoveIterSwapTestRangeIterator& t)
    requires(!IsNoExcept)
  {
    (*t.counter_)++;
    return *t.value_;
  }
  friend constexpr auto iter_move(const IterMoveIterSwapTestRangeIterator& t) noexcept
    requires IsNoExcept
  {
    (*t.counter_)++;
    return *t.value_;
  }
};
} // namespace adl

template <typename T = int*, bool IsSwappable = true, bool IsNoExcept = true>
struct IterMoveIterSwapTestRange : std::ranges::view_base {
  adl::IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept> begin_;
  adl::IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept> end_;
  constexpr IterMoveIterSwapTestRange(const T& begin, const T& end, int* counter)
      : begin_(adl::IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept>(begin, counter)),
        end_(adl::IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept>(end, counter)) {}
  constexpr adl::IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept> begin() const { return begin_; }
  constexpr adl::IterMoveIterSwapTestRangeIterator<T, IsSwappable, IsNoExcept> end() const { return end_; }
};

// Views

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>, bool IsCopyable = true>
struct MaybeCopyableAlwaysMoveableView : std::ranges::view_base {
  T begin_;
  T end_;

  constexpr explicit MaybeCopyableAlwaysMoveableView(T b, T e) : begin_(b), end_(e) {}

  constexpr MaybeCopyableAlwaysMoveableView(MaybeCopyableAlwaysMoveableView&& other)      = default;
  constexpr MaybeCopyableAlwaysMoveableView& operator=(MaybeCopyableAlwaysMoveableView&&) = default;

  constexpr MaybeCopyableAlwaysMoveableView(const MaybeCopyableAlwaysMoveableView&)
    requires(!IsCopyable)
  = delete;
  constexpr MaybeCopyableAlwaysMoveableView(const MaybeCopyableAlwaysMoveableView&)
    requires IsCopyable
  = default;

  constexpr MaybeCopyableAlwaysMoveableView& operator=(const MaybeCopyableAlwaysMoveableView&)
    requires(!IsCopyable)
  = delete;
  constexpr MaybeCopyableAlwaysMoveableView& operator=(const MaybeCopyableAlwaysMoveableView&)
    requires IsCopyable
  = default;

  constexpr T begin() const { return begin_; }
  constexpr sentinel_wrapper<T> end() const { return sentinel_wrapper<T>{end_}; }
};
static_assert(std::ranges::view<MaybeCopyableAlwaysMoveableView<cpp17_input_iterator<int*>>>);

static_assert(std::copyable<MaybeCopyableAlwaysMoveableView<cpp17_input_iterator<int*>>>);
static_assert(!std::copyable<MaybeCopyableAlwaysMoveableView<cpp17_input_iterator<int*>,
                                                             sentinel_wrapper<cpp17_input_iterator<int*>>,
                                                             false>>);

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>>
using CopyableView = MaybeCopyableAlwaysMoveableView<T, S>;

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>>
using MoveOnlyView = MaybeCopyableAlwaysMoveableView<T, S, false>;

template <std::input_iterator T, std::sentinel_for<T> S = sentinel_wrapper<T>, bool IsSized = false>
struct BasicTestView : std::ranges::view_base {
  T begin_{};
  T end_{};

  constexpr BasicTestView(T b, T e) : begin_(b), end_(e) {}

  constexpr T begin() { return begin_; }
  constexpr T begin() const { return begin_; }
  constexpr S end() { return S{end_}; }
  constexpr S end() const { return S{end_}; }

  constexpr auto size() const
    requires IsSized
  {
    return begin_ - end_;
  }
};
static_assert(std::ranges::sized_range<BasicTestView<SizedInputIterator, sentinel_wrapper<SizedInputIterator>, true>>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
