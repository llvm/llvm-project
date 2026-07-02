//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires (approximately_sized_range<Views> && ...);
// constexpr auto reserve_hint() const
//     requires (approximately_sized_range<const Views> && ...);

#include <cassert>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"

// All views use forward_iterator as their iterator so that ranges::size doesn't apply,
// making approximately_sized_range depend solely on the reserve_hint() member.

int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

struct ApproximatelySizedView : std::ranges::view_base {
  unsigned int hint_;
  constexpr explicit ApproximatelySizedView(unsigned int hint) : hint_(hint) {}
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + hint_); }
  constexpr unsigned int reserve_hint() const { return hint_; }
};

struct ApproximatelySizedNotConstView : std::ranges::view_base {
  unsigned int hint_;
  constexpr explicit ApproximatelySizedNotConstView(unsigned int hint) : hint_(hint) {}
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + hint_); }
  constexpr unsigned int reserve_hint() { return hint_; }
};

struct IntHintView : std::ranges::view_base {
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + 4); }
  constexpr int reserve_hint() const { return 4; }
};

struct UnsignedHintView : std::ranges::view_base {
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + 5); }
  constexpr unsigned int reserve_hint() const { return 5; }
};

struct NoHintView : std::ranges::view_base {
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + 3); }
};

constexpr bool test() {
  {
    // single range
    std::ranges::concat_view v(ApproximatelySizedView(8));
    assert(v.reserve_hint() == 8);
    assert(std::as_const(v).reserve_hint() == 8);
  }

  {
    // multiple ranges same type
    std::ranges::concat_view v(ApproximatelySizedView(2), ApproximatelySizedView(3));
    assert(v.reserve_hint() == 5);
    assert(std::as_const(v).reserve_hint() == 5);
  }

  {
    // const-view non-approximately-sized range
    std::ranges::concat_view v(ApproximatelySizedNotConstView(2), ApproximatelySizedView(3));
    assert(v.reserve_hint() == 5);
    static_assert(std::ranges::approximately_sized_range<decltype(v)>);
    static_assert(!std::ranges::approximately_sized_range<decltype(std::as_const(v))>);
  }

  {
    // underlying range not approximately-sized
    std::ranges::concat_view v(NoHintView{}, ApproximatelySizedView(8));
    static_assert(!std::ranges::approximately_sized_range<decltype(v)>);
    static_assert(!std::ranges::approximately_sized_range<decltype(std::as_const(v))>);
  }

  {
    // two ranges with different hint types: common type of int and unsigned int is unsigned int
    std::ranges::concat_view v(IntHintView{}, UnsignedHintView{});
    assert(v.reserve_hint() == 9);
    ASSERT_SAME_TYPE(decltype(v.reserve_hint()), unsigned int);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
