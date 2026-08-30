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
//     constexpr range_reference_v<V> inner_iterator::operator*() const;

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

#include "test_iterators.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr bool test() {
  std::vector<int> vector        = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> single_vector = {1};

  // Test `constexpr range_reference_v<V> inner_iterator::operator*() const`
  {
    // When range is general
    {
        std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
          std::ranges::subrange<Iterator, Sentinel>(
            Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))) |
          std::views::chunk(3);

      /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*chunked.begin()).begin();
      static_assert(std::same_as<int&, decltype(*it)>);
      assert(*it == 1);
    }

    // When range is smaller than chunk
    {
        std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
          std::ranges::subrange<Iterator, Sentinel>(
            Iterator(single_vector.data()), Sentinel(Iterator(single_vector.data() + single_vector.size()))) |
          std::views::chunk(3);

      /*chunk_view::__inner_iterator*/ std::input_iterator auto it = (*chunked.begin()).begin();
      static_assert(std::same_as<int&, decltype(*it)>);
      assert(*it == 1);
    }
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
