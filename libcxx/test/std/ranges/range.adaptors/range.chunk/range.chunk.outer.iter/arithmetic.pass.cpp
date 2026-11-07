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
//     friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator& i)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(const outer_iterator& i, default_sentinel_t t)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;

#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `friend constexpr difference_type operator-(default_sentinel_t t, const outer_iterator&)`
  {
    std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>>
        chunked =
            std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>(
                cpp17_input_iterator<int*>(vector.data()),
                sized_sentinel<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
            std::views::chunk(3);

    static_assert(std::same_as<decltype(std::default_sentinel - chunked.begin()),
                               typename std::ranges::iterator_t<decltype(chunked)>::difference_type>);
    assert(std::default_sentinel - chunked.begin() == 4);
    auto it = chunked.begin();
    ++it;
    ++it;
    assert(std::default_sentinel - it == 2);
    assert(it - std::default_sentinel == -2);
    ++it;
    ++it;
    assert(it == chunked.end());
    assert(std::default_sentinel - it == 0);
  }

  // Test `friend constexpr difference_type operator-(const outer_iterator&, default_sentinel_t)`
  {
    std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>>
        chunked =
            std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>(
                cpp17_input_iterator<int*>(vector.data()),
                sized_sentinel<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
            std::views::chunk(3);

    assert(chunked.begin() - std::default_sentinel == -4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
