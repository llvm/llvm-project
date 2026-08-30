//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     constexpr outer_iterator& operator++();
//     constexpr void operator++(int);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test `constexpr outer_iterator& operator++();`
  {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
      std::ranges::subrange<Iterator, Sentinel>(
        Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))) |
        std::views::chunk(2);

    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = chunked.begin();
    assert(std::ranges::equal(*++it, std::vector{3, 4}));
  }

  // Test `constexpr void operator++(int);`
  {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
      std::ranges::subrange<Iterator, Sentinel>(
        Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))) |
        std::views::chunk(2);

    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = chunked.begin();
    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(std::ranges::equal(*it, std::vector{3, 4}));
  }

  return true;
}

int main(int, char**) {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();

  static_assert(test<cpp17_input_iterator<int*>>());
  static_assert(test<cpp20_input_iterator<int*>>());
  static_assert(test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>());

  return 0;
}
