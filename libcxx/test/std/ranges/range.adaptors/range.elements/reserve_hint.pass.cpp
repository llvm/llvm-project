//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>

#include <cassert>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>

#include "types.h"

template <class T>
concept HasReserveHint = requires(T t) { t.reserve_hint(); };

struct NonApproximatelySized : std::ranges::view_base {
  using iterator = forward_iterator<std::tuple<int>*>;
  iterator begin() const;
  iterator end() const;
};
static_assert(!std::ranges::approximately_sized_range<NonApproximatelySized>);
static_assert(!std::ranges::approximately_sized_range<const NonApproximatelySized>);

static_assert(!HasReserveHint<std::ranges::elements_view<NonApproximatelySized, 0>>);
static_assert(!HasReserveHint<const std::ranges::elements_view<NonApproximatelySized, 0>>);

struct ApproximatelySizedTupleView : TupleBufferView {
  unsigned int hint_;

  template <std::size_t N>
  constexpr ApproximatelySizedTupleView(std::tuple<int> (&buf)[N], unsigned int hint)
      : TupleBufferView(buf), hint_(hint) {}

  constexpr auto begin() const { return forward_iterator<std::tuple<int>*>(buffer_); }
  constexpr auto end() const { return forward_iterator<std::tuple<int>*>(buffer_ + size_); }
  constexpr unsigned int reserve_hint() const { return hint_; }
};
static_assert(HasReserveHint<std::ranges::elements_view<ApproximatelySizedTupleView, 0>>);
static_assert(HasReserveHint<const std::ranges::elements_view<ApproximatelySizedTupleView, 0>>);

struct ApproximatelySizedNotConstTupleView : TupleBufferView {
  unsigned int hint_;

  template <std::size_t N>
  constexpr ApproximatelySizedNotConstTupleView(std::tuple<int> (&buf)[N], unsigned int hint)
      : TupleBufferView(buf), hint_(hint) {}

  constexpr auto begin() const { return forward_iterator<std::tuple<int>*>(buffer_); }
  constexpr auto end() const { return forward_iterator<std::tuple<int>*>(buffer_ + size_); }
  constexpr unsigned int reserve_hint() { return hint_; }
};
static_assert(HasReserveHint<std::ranges::elements_view<ApproximatelySizedNotConstTupleView, 0>>);
static_assert(!HasReserveHint<const std::ranges::elements_view<ApproximatelySizedNotConstTupleView, 0>>);

constexpr bool test() {
  std::tuple<int> buffer[] = {{1}, {2}, {3}};

  // non-const and const are approximately_sized
  {
    auto ev = std::views::elements<0>(ApproximatelySizedTupleView(buffer, 5));
    assert(ev.reserve_hint() == 5);
    assert(std::as_const(ev).reserve_hint() == 5);
  }

  {
    // mutable-only approximately_sized_range
    auto ev = std::views::elements<0>(ApproximatelySizedNotConstTupleView(buffer, 5));
    assert(ev.reserve_hint() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
