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
//     constexpr __outer_iterator begin();

//   V models forward_range:
//     constexpr auto begin() requires (!__simple_view<V>);
//     constexpr auto begin() const requires forward_range<const V>;

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_range.h"

constexpr bool test() {
  std::vector<int> vector       = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> empty_vector = {};

  // Test `chunk_view.begin()` when V models only input_range
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
      assert(std::ranges::equal(*it, std::vector{1, 2, 3}));
      assert(std::ranges::equal(*++it, std::vector{4, 5, 6}));
      assert(std::ranges::equal(*++it, std::vector{7, 8}));
      assert(++it == chunked.end());
    }

    // When range is empty
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                  cpp17_input_iterator<int*>(empty_vector.data()),
                  sentinel_wrapper<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(empty_vector.data()))) |
              std::views::chunk(3);

      assert(chunked.begin() == chunked.end());
    }
  }

  // Test `chunk_view.begin()` when V models forward_range
  {
    // When range is general
    {
      std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(3);
      std::ranges::chunk_view<std::ranges::ref_view<const std::vector<int>>> const_chunked =
          std::as_const(vector) | std::views::chunk(3);

      /*chunk_view::__iterator<false>*/ std::forward_iterator auto it = chunked.begin();
      assert(std::ranges::equal(*it, std::vector{1, 2, 3}));
      assert(std::ranges::equal(*++it, std::vector{4, 5, 6}));
      assert(std::ranges::equal(*++it, std::vector{7, 8}));
      assert(++it == chunked.end());
      /*chunk_view::__iterator<true>*/ std::forward_iterator auto const_it = const_chunked.begin();
      assert(std::ranges::equal(*const_it, std::vector{1, 2, 3}));
      assert(std::ranges::equal(*++const_it, std::vector{4, 5, 6}));
      assert(std::ranges::equal(*++const_it, std::vector{7, 8}));
      assert(++const_it == const_chunked.end());
    }

    // When range is empty
    {
      std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked =
          empty_vector | std::views::chunk(3);
      std::ranges::chunk_view<std::ranges::ref_view<const std::vector<int>>> const_chunked =
          std::as_const(empty_vector) | std::views::chunk(3);

      assert(chunked.begin() == chunked.end());
      assert(const_chunked.begin() == const_chunked.end());
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
