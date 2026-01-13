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

// constexpr auto operator*() const;

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>
#include <tuple>

#include "test_iterators.h"
#include "test_macros.h"

#include "../types.h"

template <class Iterator, class ValueType = int, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  using Result = std::tuple<std::iter_difference_t<EnumerateIterator>,
                            std::ranges::range_reference_t<MinimalView<Iterator, Sentinel>>>;

  std::array array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  View mv{Iterator(std::to_address(base(array.begin()))), Sentinel(Iterator(std::to_address(base(array.end()))))};
  EnumerateView ev{std::move(mv)};

  {
    auto it = ev.begin();
    for (std::size_t index = 0; index < array.size(); ++index) {
      std::same_as<Result> decltype(auto) result = *it;

      auto [resultIndex, resultValue] = result;
      assert(std::cmp_equal(index, resultIndex));
      assert(array[index] == resultValue);

      ++it;
    }

    assert(it == ev.end());
  }

  // const
  {
    auto constIt = std::as_const(ev).begin();
    for (std::size_t index = 0; index < array.size(); ++index) {
      std::same_as<Result> decltype(auto) result = *constIt;

      auto [resultIndex, resultValue] = result;
      assert(std::cmp_equal(index, resultIndex));
      assert(array[index] == resultValue);

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
