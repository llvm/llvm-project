//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range
//     constexpr value_type iterator::operator[](difference_type n) const
//       requires random_access_range<Base>;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `constexpr value_type iterator::operator[](difference_type) const`
  {
    // General
    {
      static_assert(
          std::ranges::random_access_range<
              std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>>);

      std::ranges::chunk_view<
          std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>(
                      random_access_iterator<int*>(vector.data()),
                      sentinel_wrapper<random_access_iterator<int*>>(
                          random_access_iterator<int*>(vector.data() + vector.size()))),
                  3);
      static_assert(
          std::same_as<decltype(chunked.begin()[1]), typename std::ranges::iterator_t<decltype(chunked)>::value_type>);
      assert(std::ranges::equal(chunked.begin()[1], std::vector{4, 5, 6}));
    }

    // The range is not fully divisible by the chunk size.
    {
      std::ranges::chunk_view<
          std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>(
                      random_access_iterator<int*>(vector.data()),
                      sentinel_wrapper<random_access_iterator<int*>>(
                          random_access_iterator<int*>(vector.data() + vector.size()))),
                  5);
      assert(std::ranges::equal(chunked.begin()[2], std::vector{11, 12}));
    }

    // The chunk size is 1.
    {
      std::ranges::chunk_view<
          std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>(
                      random_access_iterator<int*>(vector.data()),
                      sentinel_wrapper<random_access_iterator<int*>>(
                          random_access_iterator<int*>(vector.data() + vector.size()))),
                  1);
      assert(std::ranges::equal(chunked.begin()[2], std::vector{3}));
    }

    // The chunk size is larger than the range.
    {
      std::ranges::chunk_view<
          std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>(
                      random_access_iterator<int*>(vector.data()),
                      sentinel_wrapper<random_access_iterator<int*>>(
                          random_access_iterator<int*>(vector.data() + vector.size()))),
                  100);
      assert(std::ranges::equal(chunked.begin()[0], std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    }

    // The range has a single element.
    {
      std::vector<int> single_vector = {42};
      std::ranges::chunk_view<
          std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>(
                      random_access_iterator<int*>(single_vector.data()),
                      sentinel_wrapper<random_access_iterator<int*>>(
                          random_access_iterator<int*>(single_vector.data() + single_vector.size()))),
                  3);
      assert(std::ranges::equal(chunked.begin()[0], std::vector{42}));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
