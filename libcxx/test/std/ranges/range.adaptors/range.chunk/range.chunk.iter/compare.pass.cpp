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
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `friend constexpr bool operator==(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator==(const iterator&, default_sentinel_t)`
  {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
      std::ranges::subrange<Iterator, Sentinel>(
        Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))),
      3);

    auto first  = chunked.begin();
    auto second = first;
    assert(first == second);
    ++second;
    assert(first != second);
    assert(first != std::default_sentinel);
    std::ranges::advance(second, 3);
    assert(second == std::default_sentinel);
  }

  // Test `friend constexpr bool operator<(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator>(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator<=(const iterator&, const iterator&)`
  // Test `friend constexpr bool operator>=(const iterator&, const iterator&)`
  if constexpr (std::random_access_iterator<Iterator>) {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
        std::ranges::subrange<Iterator, Sentinel>(
            Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))),
        3);

    assert(!(chunked.begin() < chunked.begin()));
    assert(chunked.begin() < chunked.begin() + 1);
    assert(!(chunked.begin() + 1 < chunked.begin()));
    assert(!(chunked.begin() > chunked.begin()));
    assert(!(chunked.begin() > chunked.begin() + 1));
    assert(chunked.begin() + 1 > chunked.begin());
    assert(chunked.begin() <= chunked.begin());
    assert(chunked.begin() <= chunked.begin() + 1);
    assert(!(chunked.begin() + 1 <= chunked.begin()));
    assert(chunked.begin() + 1 >= chunked.begin());
    assert(chunked.begin() + 1 >= chunked.begin() + 1);
    assert(!(chunked.begin() >= chunked.begin() + 1));
  }

  // Test `friend constexpr auto operator<=>(const iterator&, const iterator&)`
  if constexpr (std::random_access_iterator<Iterator> && std::three_way_comparable<Iterator>) {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
        std::ranges::subrange<Iterator, Sentinel>(
            Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))),
        3);

    assert((chunked.begin() <=> chunked.begin() + 1) == std::strong_ordering::less);
    assert((chunked.begin() <=> chunked.begin()) == std::strong_ordering::equal);
    assert((chunked.begin() + 1 <=> chunked.begin()) == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test<int*, int*>();

  static_assert(test<forward_iterator<int*>>());
  static_assert(test<bidirectional_iterator<int*>>());
  static_assert(test<random_access_iterator<int*>>());
  static_assert(test<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>());
  static_assert(test<int*, int*>());

  return 0;
}
