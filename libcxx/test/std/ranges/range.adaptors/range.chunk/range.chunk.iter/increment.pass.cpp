//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range:
//     constexpr iterator& operator++();
//     constexpr iterator operator++(int);
//     constexpr iterator& operator+=(difference_type)
//       requires random_access_range<Base>;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test `constexpr iterator& operator++()`
  // Test `constexpr iterator operator++(int)`
  {
    std::ranges::chunk_view<
        std::ranges::subrange<forward_iterator<int*>, forward_iterator<int*>, std::ranges::subrange_kind::sized>>
        chunked(
            std::ranges::subrange<forward_iterator<int*>, forward_iterator<int*>, std::ranges::subrange_kind::sized>(
                forward_iterator<int*>(vector.data()),
                forward_iterator<int*>(vector.data() + vector.size()),
                vector.size()),
            2);
    auto it                                                                         = chunked.begin();
    std::same_as<std::ranges::iterator_t<decltype(chunked)>&> decltype(auto) result = ++it;
    assert(&result == &it);
    assert(std::ranges::equal(*it, std::vector{3, 4}));
    std::same_as<std::ranges::iterator_t<decltype(chunked)>> decltype(auto) it2 = it++;
    assert(std::ranges::equal(*it, std::vector{5, 6}));
    assert(std::ranges::equal(*it2, std::vector{3, 4}));
  }

  // Test `constexpr iterator& operator+=(difference_type)`
  {
    std::ranges::chunk_view< std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>> chunked(
        std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
            random_access_iterator<int*>(vector.data()),
            random_access_iterator<int*>(vector.data() + vector.size()),
            vector.size()),
        2);
    auto it = chunked.begin();
    it += 1;
    assert(std::ranges::equal(*it, std::vector{3, 4}));
    it += 2;
    assert(std::ranges::equal(*it, std::vector{7, 8}));
    it += -1;
    assert(std::ranges::equal(*it, std::vector{5, 6}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
