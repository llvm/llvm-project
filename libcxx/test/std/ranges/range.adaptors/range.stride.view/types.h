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

#include "__iterator/concepts.h"
#include "test_iterators.h"
#include "test_range.h"

// Concepts

template <typename Iter>
concept IterDifferable = requires(Iter& t) { t - t; };

// Iterators

template <class Derived, std::input_iterator Iter = int*, bool IsSized = false>
  requires((!IsSized) || (IsSized && IterDifferable<Iter>))
struct InputIterBase {
  using iterator_concept  = std::input_iterator_tag;
  using iterator_category = std::input_iterator_tag;
  using value_type        = typename std::iterator_traits<Iter>::value_type;
  using difference_type   = typename std::iterator_traits<Iter>::difference_type;

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
    requires IsSized
  {
    return left.value_ - right.value_;
  }
};

struct UnsizedInputIterator : InputIterBase<UnsizedInputIterator> {};
static_assert(std::input_iterator<UnsizedInputIterator>);
static_assert(!std::sized_sentinel_for<UnsizedInputIterator, UnsizedInputIterator>);

struct SizedInputIterator : InputIterBase<SizedInputIterator, int*, true> {
  using InputIterBase::InputIterBase;
};
static_assert(std::input_iterator<SizedInputIterator>);
static_assert(std::sized_sentinel_for<SizedInputIterator, SizedInputIterator>);

// Put IterMoveIterSwapTestRangeIterator in a namespace to test ADL of CPOs iter_swap and iter_move
// (see iter_swap.pass.cpp and iter_move.pass.cpp).
namespace adl {
template <std::input_iterator Iter = int*, bool IsIterSwappable = true, bool IsNoExceptIterMoveable = true>
struct IterMoveIterSwapTestRangeIterator
    : InputIterBase<IterMoveIterSwapTestRangeIterator<Iter, IsIterSwappable, IsNoExceptIterMoveable>, Iter, false> {
  int* counter_{nullptr};

  using InputIterBase<IterMoveIterSwapTestRangeIterator<Iter, IsIterSwappable, IsNoExceptIterMoveable>, Iter, false>::
      InputIterBase;

  constexpr IterMoveIterSwapTestRangeIterator(Iter value, int* counter)
      : InputIterBase<IterMoveIterSwapTestRangeIterator<Iter, IsIterSwappable, IsNoExceptIterMoveable>, Iter, false>(
            value),
        counter_(counter) {}

  friend constexpr void iter_swap(IterMoveIterSwapTestRangeIterator t, IterMoveIterSwapTestRangeIterator u) noexcept
    requires IsIterSwappable && IsNoExceptIterMoveable
  {
    (*t.counter_)++;
    (*u.counter_)++;
    std::swap(*t.value_, *u.value_);
  }

  friend constexpr void iter_swap(IterMoveIterSwapTestRangeIterator t, IterMoveIterSwapTestRangeIterator u)
    requires IsIterSwappable && (!IsNoExceptIterMoveable)
  {
    (*t.counter_)++;
    (*u.counter_)++;
    std::swap(*t.value_, *u.value_);
  }

  friend constexpr auto iter_move(const IterMoveIterSwapTestRangeIterator& t)
    requires(!IsNoExceptIterMoveable)
  {
    (*t.counter_)++;
    return *t.value_;
  }
  friend constexpr auto iter_move(const IterMoveIterSwapTestRangeIterator& t) noexcept
    requires IsNoExceptIterMoveable
  {
    (*t.counter_)++;
    return *t.value_;
  }
};
} // namespace adl

template <typename Iter = int*, bool IsSwappable = true, bool IsNoExcept = true>
struct IterMoveIterSwapTestRange : std::ranges::view_base {
  adl::IterMoveIterSwapTestRangeIterator<Iter, IsSwappable, IsNoExcept> begin_;
  adl::IterMoveIterSwapTestRangeIterator<Iter, IsSwappable, IsNoExcept> end_;
  constexpr IterMoveIterSwapTestRange(const Iter& begin, const Iter& end, int* counter)
      : begin_(adl::IterMoveIterSwapTestRangeIterator<Iter, IsSwappable, IsNoExcept>(begin, counter)),
        end_(adl::IterMoveIterSwapTestRangeIterator<Iter, IsSwappable, IsNoExcept>(end, counter)) {}
  constexpr adl::IterMoveIterSwapTestRangeIterator<Iter, IsSwappable, IsNoExcept> begin() const { return begin_; }
  constexpr adl::IterMoveIterSwapTestRangeIterator<Iter, IsSwappable, IsNoExcept> end() const { return end_; }
};

// Views

template <bool View>
struct ViewOrRange {};

template <>
struct ViewOrRange<true> : std::ranges::view_base {};

template <std::input_iterator Iter,
          std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>,
          bool IsSized                 = false,
          bool IsView                  = false>
  requires((!IsSized) || (IsSized && IterDifferable<Iter>))
struct BasicTestViewOrRange : ViewOrRange<IsView> {
  Iter begin_{};
  Iter end_{};

  constexpr BasicTestViewOrRange(Iter b, Iter e) : begin_(b), end_(e) {}

  constexpr Iter begin() { return begin_; }
  constexpr Iter begin() const { return begin_; }
  constexpr Sent end() { return Sent{end_}; }
  constexpr Sent end() const { return Sent{end_}; }

  constexpr auto size() const
    requires IsSized
  {
    return begin_ - end_;
  }
};

template <std::input_iterator Iter, std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>, bool IsSized = false>
  requires((!IsSized) || (IsSized && IterDifferable<Iter>))
using BasicTestView = BasicTestViewOrRange<Iter, Sent, IsSized, true>;

template <std::input_iterator Iter, std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>, bool IsCopyable = true>
struct MaybeCopyableAlwaysMoveableView : std::ranges::view_base {
  Iter begin_;
  Iter end_;

  constexpr explicit MaybeCopyableAlwaysMoveableView(Iter b, Iter e) : begin_(b), end_(e) {}

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

  constexpr Iter begin() const { return begin_; }
  constexpr Sent end() const { return Sent{end_}; }
};
static_assert(std::ranges::view<MaybeCopyableAlwaysMoveableView<cpp17_input_iterator<int*>>>);
static_assert(std::ranges::view<MaybeCopyableAlwaysMoveableView<cpp17_input_iterator<int*>,
                                                                sentinel_wrapper<cpp17_input_iterator<int*>>,
                                                                false>>);

static_assert(std::copyable<MaybeCopyableAlwaysMoveableView<cpp17_input_iterator<int*>>>);
template <std::input_iterator Iter, std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>>
using CopyableView = MaybeCopyableAlwaysMoveableView<Iter, Sent>;
static_assert(std::copyable<CopyableView<cpp17_input_iterator<int*>>>);

template <std::input_iterator Iter, std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>>
using MoveOnlyView = MaybeCopyableAlwaysMoveableView<Iter, Sent, false>;
static_assert(!std::copyable<MoveOnlyView<cpp17_input_iterator<int*>>>);

// Don't move/hold the iterator itself, copy/hold the base
// of that iterator and reconstruct the iterator on demand.
// May result in aliasing (if, e.g., Iterator is an iterator
// over int *).
template <class Iter, std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>>
struct ViewOverNonCopyableIterator : std::ranges::view_base {
  constexpr explicit ViewOverNonCopyableIterator(Iter it, Sent sent) : it_(base(it)), sent_(base(sent)) {}

  ViewOverNonCopyableIterator(ViewOverNonCopyableIterator&&)            = default;
  ViewOverNonCopyableIterator& operator=(ViewOverNonCopyableIterator&&) = default;

  constexpr Iter begin() const { return Iter(it_); }
  constexpr Sent end() const { return Sent(sent_); }

private:
  decltype(base(std::declval<Iter>())) it_;
  decltype(base(std::declval<Sent>())) sent_;
};

// Ranges

template <std::input_iterator Iter, std::sentinel_for<Iter> Sent = sentinel_wrapper<Iter>, bool IsSized = false>
  requires((!IsSized) || (IsSized && IterDifferable<Iter>))
using BasicTestRange = BasicTestViewOrRange<Iter, Sent, IsSized, false>;

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
