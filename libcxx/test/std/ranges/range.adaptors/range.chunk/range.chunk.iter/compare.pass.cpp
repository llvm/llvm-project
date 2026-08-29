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
//     friend constexpr bool operator==(const iterator& x, const iterator& y);
//     friend constexpr bool operator==(const iterator& x, default_sentinel_t);
//     friend constexpr bool operator<(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr bool operator>(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr bool operator<=(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr bool operator>=(const iterator& x, const iterator& y)
//       requires random_access_range<Base>;
//     friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//       requires random_access_range<Base> &&
//                three_way_comparable<iterator_t<Base>>;

#include <cassert>
#include <compare>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `friend constexpr bool operator==(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator==(const iterator&, default_sentinel_t)`
  {
    std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>> chunked(
        std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
            random_access_iterator<int*>(vector.data()), random_access_iterator<int*>(vector.data() + vector.size())),
        3);

    assert(chunked.begin() == chunked.begin());
    assert(chunked.begin() + 1 == chunked.end() - 3);
    assert(chunked.begin() + 4 == chunked.end());
    assert(chunked.end() == std::default_sentinel);
  }

  // Test `friend constexpr bool operator<(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator>(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator<=(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator>=(const iterator&, const iterator&)`
  {
    std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>> chunked(
        std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
            random_access_iterator<int*>(vector.data()), random_access_iterator<int*>(vector.data() + vector.size())),
        3);

    assert(chunked.begin() < chunked.end());
    assert(chunked.begin() < chunked.begin() + 1);
    assert(chunked.begin() + 1 > chunked.begin());
    assert(chunked.end() > chunked.begin());
    assert(chunked.begin() <= chunked.begin());
    assert(chunked.begin() <= chunked.begin() + 1);
    assert(chunked.begin() + 1 >= chunked.begin());
    assert(chunked.begin() + 1 >= chunked.begin() + 1);
  }

  // Test `friend constexpr auto operator<=>(const iterator&, const iterator&)`
  {
    std::ranges::chunk_view<std::ranges::subrange<int*, int*>> chunked(
        std::ranges::subrange<int*, int*>(vector.data(), vector.data() + vector.size()), 3);

    assert((chunked.begin() <=> chunked.end()) == std::strong_ordering::less);
    assert((chunked.begin() <=> chunked.begin() + 1) == std::strong_ordering::less);
    assert((chunked.begin() <=> chunked.begin()) == std::strong_ordering::equal);
    assert((chunked.begin() + 1 <=> chunked.end() - 3) == std::strong_ordering::equal);
    assert((chunked.end() <=> chunked.begin()) == std::strong_ordering::greater);
    assert((chunked.begin() + 1 <=> chunked.begin()) == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
