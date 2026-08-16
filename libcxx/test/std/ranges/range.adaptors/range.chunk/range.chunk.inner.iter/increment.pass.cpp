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
//     constexpr inner_iterator& operator++();
//     constexpr void operator++(int);

#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  // Test `constexpr inner_iterator& operator++()`
  {
    std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};
    std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
        chunked =
            std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                cpp17_input_iterator<int*>(vector.data()),
                sentinel_wrapper<cpp17_input_iterator<int*>>(
                    cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
            std::views::chunk(2);

    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*chunked.begin()).begin();
    assert(*++it == 2);
  }

  // Test `constexpr void operator++(int)`
  {
    std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};
    std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
        chunked =
            std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                cpp17_input_iterator<int*>(vector.data()),
                sentinel_wrapper<cpp17_input_iterator<int*>>(
                    cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
            std::views::chunk(2);

    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*chunked.begin()).begin();
    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(*it == 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
