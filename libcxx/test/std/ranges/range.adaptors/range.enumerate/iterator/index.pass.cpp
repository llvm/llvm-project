//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// constexpr difference_type index() const noexcept;

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <utility>
#include <tuple>

#include "test_iterators.h"
#include "test_macros.h"

#include "../types.h"

template <class Iterator, class ValueType = int, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View          = MinimalView<Iterator, Sentinel>;
  using EnumerateView = std::ranges::enumerate_view<View>;

  std::array array{94, 82, 47};

  View mv{Iterator(std::to_address(base(array.begin()))), Sentinel(Iterator(std::to_address(base(array.end()))))};
  EnumerateView ev(std::move(mv));

  using DifferenceT = std::ranges::range_difference_t<decltype(ev)>;

  {
    auto it = ev.begin();

    std::same_as<DifferenceT> decltype(auto) index = it.index();

    assert(std::cmp_equal(0, index));
    ++it;
    assert(std::cmp_equal(1, it.index()));
    ++it;
    assert(std::cmp_equal(2, it.index()));
    ++it;

    assert(it == ev.end());

    static_assert(noexcept(it.index()));
  }

  // const
  {
    auto it = std::as_const(ev).begin();

    std::same_as<DifferenceT> decltype(auto) index = it.index();

    assert(std::cmp_equal(0, index));
    ++it;
    assert(std::cmp_equal(1, it.index()));
    ++it;
    assert(std::cmp_equal(2, it.index()));
    ++it;

    assert(it == ev.end());

    static_assert(noexcept(it.index()));
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  test<cpp17_input_iterator<const int*>, const int>();
  test<cpp20_input_iterator<const int*>, const int>();
  test<forward_iterator<const int*>, const int>();
  test<bidirectional_iterator<const int*>, const int>();
  test<random_access_iterator<const int*>, const int>();
  test<contiguous_iterator<const int*>, const int>();
  test<const int*, const int>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
