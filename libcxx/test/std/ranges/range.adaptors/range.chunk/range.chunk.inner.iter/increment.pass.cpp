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
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test `constexpr inner_iterator& operator++()`
  {
    std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
        chunked =
            std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                cpp17_input_iterator<int*>(vector.data()),
                sentinel_wrapper<cpp17_input_iterator<int*>>(
                    cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
            std::views::chunk(3);

    /*chunk_view::__outer_iterator*/ auto outer = chunked.begin();
    /*chunk_view::__inner_iterator*/ auto inner = (*outer).begin();
    assert(*++inner == 2);
    assert(*++inner == 3);
    assert(++inner == std::default_sentinel);
    ++outer;
    inner = (*outer).begin();
    assert(*++inner == 5);
    assert(*++inner == 6);
    assert(++inner == std::default_sentinel);
    ++outer;
    inner = (*outer).begin();
    assert(*++inner == 8);
    assert(++inner == std::default_sentinel);
  }

  // Test `constexpr void operator++(int)`
  {
    std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
        chunked =
            std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                cpp17_input_iterator<int*>(vector.data()),
                sentinel_wrapper<cpp17_input_iterator<int*>>(
                    cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
            std::views::chunk(3);

    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*chunked.begin()).begin();
    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(*it == 2);
    it++;
    assert(*it == 3);
    it++;
    assert(it == std::default_sentinel);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
