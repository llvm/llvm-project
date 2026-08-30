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
//     constexpr const iterator_t<V> base() const&;

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

#include "test_iterators.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr bool test() {
  std::vector<int> vector        = {1, 2, 3, 4};
  std::vector<int> single_vector = {1};

  // Test `constexpr const iterator_t<V> base() const&`
  {
    // When range is general
    {
        std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
          std::ranges::subrange<Iterator, Sentinel>(
            Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))),
          2);
      auto outer = chunked.begin();
      auto inner = (*outer).begin();
      static_assert(std::same_as<const Iterator, decltype(inner.base())>);
      assert(*inner.base() == 1);
      ++inner;
      assert(*inner.base() == 2);
    }

    // When range is single
    {
        std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
          std::ranges::subrange<Iterator, Sentinel>(
            Iterator(single_vector.data()), Sentinel(Iterator(single_vector.data() + single_vector.size()))),
          2);
      auto outer = chunked.begin();
      auto inner = (*outer).begin();
      static_assert(std::same_as<const Iterator, decltype(inner.base())>);
      assert(*inner.base() == 1);
    }
  }

  return true;
}

int main(int, char**) {
  test<cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();

  static_assert(test<cpp17_input_iterator<int*>>());
  static_assert(test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>());

  return 0;
}
