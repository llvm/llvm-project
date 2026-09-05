//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<View>;
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const View>;

#include <cassert>
#include <cstddef>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"

int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

struct NonApproximatelySizedView : std::ranges::view_base {
  using iterator = forward_iterator<int*>;
  iterator begin() const;
  iterator end() const;
};

struct ApproximatelySizedView : std::ranges::view_base {
  unsigned int size_;
  constexpr explicit ApproximatelySizedView(unsigned int size) : size_(size) {}
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + size_); }
  constexpr unsigned int reserve_hint() const { return size_; }
};

struct ApproximatelySizedNotConstView : std::ranges::view_base {
  unsigned int size_;
  constexpr explicit ApproximatelySizedNotConstView(unsigned int size) : size_(size) {}
  constexpr auto begin() const { return forward_iterator<int*>(buffer); }
  constexpr auto end() const { return forward_iterator<int*>(buffer + size_); }
  constexpr unsigned int reserve_hint() { return size_; }
};

template <std::size_t N>
constexpr void test() {
  {
    // Test with different values of N for an approximately sized view
    std::ranges::adjacent_view<ApproximatelySizedView, N> v(ApproximatelySizedView(5));
    static_assert(std::ranges::approximately_sized_range<decltype(v)>);
    static_assert(std::ranges::approximately_sized_range<const decltype(v)>);

    auto expected_hint = 5 - (N - 1);
    assert(v.reserve_hint() == expected_hint);
    assert(std::as_const(v).reserve_hint() == expected_hint);
  }
  {
    // Test with different values of N for a non-const approximately sized view
    std::ranges::adjacent_view<ApproximatelySizedNotConstView, N> v(ApproximatelySizedNotConstView(5));
    static_assert(std::ranges::approximately_sized_range<decltype(v)>);
    static_assert(!std::ranges::approximately_sized_range<const decltype(v)>);

    auto expected_hint = 5 - (N - 1);
    assert(v.reserve_hint() == expected_hint);
  }
  {
    // empty range
    std::ranges::adjacent_view<ApproximatelySizedView, N> v(ApproximatelySizedView(0));
    static_assert(std::ranges::approximately_sized_range<decltype(v)>);
    static_assert(std::ranges::approximately_sized_range<const decltype(v)>);

    assert(v.reserve_hint() == 0);
    assert(std::as_const(v).reserve_hint() == 0);
  }
  {
    // N greater than range size
    if constexpr (N > 2) {
      std::ranges::adjacent_view<ApproximatelySizedView, N> v(ApproximatelySizedView(2));
      static_assert(std::ranges::approximately_sized_range<decltype(v)>);
      static_assert(std::ranges::approximately_sized_range<const decltype(v)>);
      assert(v.reserve_hint() == 0);
      assert(std::as_const(v).reserve_hint() == 0);
    }
  }
}

constexpr bool test() {
  // non-approximately-sized range has no reserve_hint
  static_assert(!std::ranges::approximately_sized_range<std::ranges::adjacent_view<NonApproximatelySizedView, 2>>);
  static_assert(
      !std::ranges::approximately_sized_range<const std::ranges::adjacent_view<NonApproximatelySizedView, 2>>);

  test<1>();
  test<2>();
  test<3>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
