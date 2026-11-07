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
//     friend constexpr iterator operator+(const iterator& i, difference_type n)
//       requires random_access_range<Base>;
//     friend constexpr iterator operator+(difference_type n, const iterator& i)
//       requires random_access_range<Base>;
//     friend constexpr iterator operator-(const iterator& i, difference_type n)
//       requires random_access_range<Base>;
//     friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//       requires sized_sentinel_for<iterator_t<Base>, iterator_t<Base>>;
//     friend constexpr difference_type operator-(default_sentinel_t y, const iterator& x)
//       requires sized_sentinel_for<sentinel_t<Base>, iterator_t<Base>>;
//     friend constexpr difference_type operator-(const iterator& x, default_sentinel_t y)
//       requires sized_sentinel_for<sentinel_t<Base>, iterator_t<Base>>;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `friend constexpr iterator operator+(const iterator&, difference_type)`
  // Test `friend constexpr iterator operator+(different_type, const iterator&)`
  // Test `friend constexpr iterator operator-(const iterator&, difference_type)`
  // Test `friend constexpr difference_type operator-(const iterator&, const iterator&)`
  // Test `friend constexpr difference_type operator-(default_sentinel_t, const iterator&)`
  // Test `friend constexpr difference_type operator-(const iterator&, default_sentinel_t)`
  {
    static_assert(requires(
        std::ranges::iterator_t<std::ranges::chunk_view<
            std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>> t,
        std::iter_difference_t<std::ranges::iterator_t<std::ranges::chunk_view<
            std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>>> n) {
      t + n;
      n + t;
      t - n;
      t - t;
      std::default_sentinel - t;
      t - std::default_sentinel;
    });

    // General.
    {
      std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
                      random_access_iterator<int*>(vector.data()),
                      random_access_iterator<int*>(vector.data() + vector.size())),
                  3);

      assert(chunked.begin() + 4 == chunked.end());
      assert(4 + chunked.begin() == chunked.end());
      assert(chunked.end() - 4 == chunked.begin());
      assert(chunked.end() - chunked.begin() == 4);
      assert(std::default_sentinel - chunked.begin() == 4);
      assert(chunked.begin() - std::default_sentinel == -4);
    }

    // The chunk size is 1.
    {
      std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
                      random_access_iterator<int*>(vector.data()),
                      random_access_iterator<int*>(vector.data() + vector.size())),
                  1);
      assert(chunked.begin() + 12 == chunked.end());
      assert(12 + chunked.begin() == chunked.end());
      assert(chunked.end() - 12 == chunked.begin());
      assert(chunked.end() - chunked.begin() == 12);
      assert(std::default_sentinel - chunked.begin() == 12);
      assert(chunked.begin() - std::default_sentinel == -12);
    }

    // The chunk size is larger than the range.
    {
      std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
                      random_access_iterator<int*>(vector.data()),
                      random_access_iterator<int*>(vector.data() + vector.size())),
                  100);
      assert(chunked.begin() + 1 == chunked.end());
      assert(1 + chunked.begin() == chunked.end());
      assert(chunked.end() - 1 == chunked.begin());
      assert(chunked.end() - chunked.begin() == 1);
      assert(std::default_sentinel - chunked.begin() == 1);
      assert(chunked.begin() - std::default_sentinel == -1);
    }

    // The range has a single element.
    {
      std::vector<int> single_vector = {1};
      std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
                      random_access_iterator<int*>(single_vector.data()),
                      random_access_iterator<int*>(single_vector.data() + single_vector.size())),
                  3);
      assert(chunked.begin() + 1 == chunked.end());
      assert(1 + chunked.begin() == chunked.end());
      assert(chunked.end() - 1 == chunked.begin());
      assert(chunked.end() - chunked.begin() == 1);
      assert(std::default_sentinel - chunked.begin() == 1);
      assert(chunked.begin() - std::default_sentinel == -1);
    }

    // The range is not fully divisible by the chunk size.
    {
      std::ranges::chunk_view<std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>>
          chunked(std::ranges::subrange<random_access_iterator<int*>, random_access_iterator<int*>>(
                      random_access_iterator<int*>(vector.data()),
                      random_access_iterator<int*>(vector.data() + vector.size())),
                  5);
      assert(chunked.begin() + 3 == chunked.end());
      assert(3 + chunked.begin() == chunked.end());
      assert(chunked.end() - 3 == chunked.begin());
      assert(chunked.end() - chunked.begin() == 3);
      assert(std::default_sentinel - chunked.begin() == 3);
      assert(chunked.begin() - std::default_sentinel == -3);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
