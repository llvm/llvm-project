//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

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

  std::array array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  View view{Iterator(array.begin()), Sentinel(Iterator(array.end()))};
  EnumerateView ev(std::move(view));

  {
    auto it = ev.begin();
    ASSERT_NOEXCEPT(it.index());

    static_assert(std::same_as<typename decltype(it)::difference_type, decltype(it.index())>);
    for (std::size_t index = 0; index < array.size(); ++index) {
      assert(std::cmp_equal(index, it.index()));

      ++it;
    }

    assert(it == ev.end());
  }

  // const
  {
    auto constIt = std::as_const(ev).begin();
    ASSERT_NOEXCEPT(constIt.index());

    static_assert(std::same_as<typename decltype(constIt)::difference_type, decltype(constIt.index())>);
    for (std::size_t index = 0; index < array.size(); ++index) {
      assert(std::cmp_equal(index, constIt.index()));

      ++constIt;
    }

    assert(constIt == ev.end());
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

  test<cpp17_input_iterator<int const*>, int const>();
  test<cpp20_input_iterator<int const*>, int const>();
  test<forward_iterator<int const*>, int const>();
  test<bidirectional_iterator<int const*>, int const>();
  test<random_access_iterator<int const*>, int const>();
  test<contiguous_iterator<int const*>, int const>();
  test<int const*, int const>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
