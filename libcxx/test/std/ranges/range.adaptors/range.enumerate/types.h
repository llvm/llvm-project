//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_H

#include <cstddef>
#include <ranges>
#include <tuple>

#include "test_iterators.h"
#include "test_macros.h"

// Types

template <typename T, typename DifferenceT = std::ptrdiff_t>
using ValueType = std::tuple<DifferenceT, T>;

struct RangeView : std::ranges::view_base {
  using Iterator = cpp20_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;

  constexpr explicit RangeView(int* b, int* e) : begin_(b), end_(e) {}
  constexpr RangeView(RangeView const& other) : begin_(other.begin_), end_(other.end_), wasCopyInitialized(true) {}
  constexpr RangeView(RangeView&& other) : begin_(other.begin_), end_(other.end_), wasMoveInitialized(true) {}
  RangeView& operator=(RangeView const&) = default;
  RangeView& operator=(RangeView&&)      = default;

  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_;
  int* end_;

  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

LIBCPP_STATIC_ASSERT(std::ranges::__range_with_movable_references<RangeView>);
static_assert(std::ranges::view<RangeView>);

struct MinimalDefaultConstructedView : std::ranges::view_base {
  MinimalDefaultConstructedView() = default;

  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};

static_assert(std::ranges::view<MinimalDefaultConstructedView>);

template <class Iterator, class Sentinel>
struct MinimalView : std::ranges::view_base {
  constexpr explicit MinimalView(Iterator it, Sentinel sent) : it_(base(std::move(it))), sent_(base(std::move(sent))) {}

  MinimalView(MinimalView&&)            = default;
  MinimalView& operator=(MinimalView&&) = default;

  constexpr Iterator begin() const { return Iterator(it_); }
  constexpr Sentinel end() const { return Sentinel(sent_); }

private:
  decltype(base(std::declval<Iterator>())) it_;
  decltype(base(std::declval<Sentinel>())) sent_;
};

static_assert(std::ranges::view<MinimalView<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>);

struct NotInvocable {};

static_assert(!std::invocable<NotInvocable>);

struct NotAView {};

static_assert(!std::ranges::view<NotAView>);

struct NotAViewRange {
  using Iterator = cpp20_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;

  NotAViewRange() = default;
  constexpr explicit NotAViewRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr NotAViewRange(NotAViewRange const& other) = default;
  constexpr NotAViewRange(NotAViewRange&& other)      = default;
  NotAViewRange& operator=(NotAViewRange const&)      = default;
  NotAViewRange& operator=(NotAViewRange&&)           = default;

  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_;
  int* end_;
};

static_assert(std::ranges::range<NotAViewRange>);
static_assert(!std::ranges::view<NotAViewRange>);

template <bool IsNoexcept>
class MaybeNoexceptIterMoveInputIterator {
  int* it_;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = typename std::iterator_traits<int*>::difference_type;
  using pointer           = int*;
  using reference         = int&;

  MaybeNoexceptIterMoveInputIterator() = default;
  explicit constexpr MaybeNoexceptIterMoveInputIterator(int* it) : it_(it) {}

  friend constexpr decltype(auto) iter_move(const MaybeNoexceptIterMoveInputIterator& it) noexcept(IsNoexcept) {
    return std::ranges::iter_move(it.it_);
  }

  friend constexpr int* base(const MaybeNoexceptIterMoveInputIterator& i) { return i.it_; }

  constexpr reference operator*() const { return *it_; }
  constexpr MaybeNoexceptIterMoveInputIterator& operator++() {
    ++it_;
    return *this;
  }
  constexpr MaybeNoexceptIterMoveInputIterator operator++(int) {
    MaybeNoexceptIterMoveInputIterator tmp(*this);
    ++(*this);
    return tmp;
  }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_H
