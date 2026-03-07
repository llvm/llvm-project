//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range
//     constexpr outer_iterator& operator++();
//     constexpr void operator++(int);
//     constexpr inner_iterator& operator++();
//     constexpr void operator++(int);

//   V models forward_range
//     constexpr iterator& operator++();
//     constexpr iterator operator++(int);
//     constexpr iterator& operator+=(difference_type)
//       requires random_access_range<Base>;

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_range.h"
#include "../types.h"

constexpr bool test() {
  std::vector<int> vector                                                  = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(2);
  std::ranges::chunk_view<input_span<int>> input_chunked = input_span<int>(vector) | std::views::chunk(2);

  // Test `constexpr outer_iterator& operator++();`
  {
    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = input_chunked.begin();
    assert(std::ranges::equal(*++it, std::vector{3, 4}));
  }

  // Test `constexpr void operator++(int);`
  {
    /*chunk_view::__outer_iterator*/ std::input_iterator auto it = input_chunked.begin();
    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(std::ranges::equal(*it, std::vector{3, 4}));
  }

  // Test `constexpr inner_iterator& operator++();`
  {
    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*input_chunked.begin()).begin();
    assert(*++it == 2);
  }

  // Test `constexpr inner_iterator& operator++();`
  {
    /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*input_chunked.begin()).begin();
    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(*it == 2);
  }

  // Test `constexpr iterator& operator++();`
  {
    /*chunk_view::__iterator*/ std::forward_iterator auto it = chunked.begin();
    assert(std::ranges::equal(*++it, std::vector{3, 4}));
  }

  // Test `constexpr iterator operator++(int)`
  {
    /*chunk_view::__iterator*/ std::forward_iterator auto it = chunked.begin();
    it++;
    assert(std::ranges::equal(*it, std::vector{3, 4}));
  }

  // Test `constexpr iterator& operator+=(difference_type)`
  {
    /*chunk_view::__iterator*/ std::random_access_iterator auto it = chunked.begin();
    it += 3;
    assert(std::ranges::equal(*it, std::vector{7, 8}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
