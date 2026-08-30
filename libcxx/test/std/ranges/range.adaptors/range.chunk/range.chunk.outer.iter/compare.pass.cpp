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
//     friend constexpr bool operator==(const outer_iterator& x, default_sentinel_t);

#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `friend constexpr bool operator==(const outer_iterator& x, default_sentinel_t)`
  {
    // When range is general
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                  cpp17_input_iterator<int*>(vector.data()),
                  sentinel_wrapper<cpp17_input_iterator<int*>>(
                      cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
              std::views::chunk(3);

      /*chunk_view::__outer_iterator*/ std::input_iterator auto it = chunked.begin();
      assert(it != std::default_sentinel);
      std::ranges::advance(it, 4);
      assert(it == std::default_sentinel);
    }

    // When chunk size is one
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                  cpp17_input_iterator<int*>(vector.data()),
                  sentinel_wrapper<cpp17_input_iterator<int*>>(
                      cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
              std::views::chunk(1);

      /*chunk_view::__outer_iterator*/ std::input_iterator auto it = chunked.begin();
      assert(it != std::default_sentinel);
      std::ranges::advance(it, 12);
      assert(it == std::default_sentinel);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
