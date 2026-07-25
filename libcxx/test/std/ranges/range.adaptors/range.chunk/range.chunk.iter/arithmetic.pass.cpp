//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// V models forward_range:
//   friend constexpr iterator operator+(const iterator& i, difference_type n)
//     requires random_access_range<Base>;
//   friend constexpr iterator operator+(difference_type n, const iterator& i)
//     requires random_access_range<Base>;
//   friend constexpr iterator operator-(const iterator& i, difference_type n)
//     requires random_access_range<Base>;
//   friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//     requires sized_sentinel_for<iterator_t<Base>, iterator_t<Base>>;
//   friend constexpr difference_type operator-(default_sentinel_t y, const iterator& x)
//     requires sized_sentinel_for<sentinel_t<Base>, iterator_t<Base>>;
//   friend constexpr difference_type operator-(const iterator& x, default_sentinel_t y)
//     requires sized_sentinel_for<sentinel_t<Base>, iterator_t<Base>>;

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

constexpr bool test() {
  std::vector<int> vector                                                  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(3);

  // Test `friend constexpr iterator operator+(const iterator& i, difference_type n)`
  {
    assert(chunked.begin() + 4 == chunked.end());
  }

  // Test `friend constexpr iterator operator+(difference_type n, const iterator& i)`
  {
    assert(4 + chunked.begin() == chunked.end());
  }

  // Test `friend constexpr iterator operator-(const iterator& i, difference_type n)`
  {
    assert(chunked.end() - 4 == chunked.begin());
  }

  // Test `friend constexpr difference_type operator-(const iterator& x, const iterator& y)`
  {
    assert(chunked.end() - chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(default_sentinel_t y, const iterator& x)`
  {
    assert(std::default_sentinel - chunked.begin() == 4);
  }

  // Test `friend constexpr difference_type operator-(const iterator& x, default_sentinel_t y)`
  {
    assert(chunked.begin() - std::default_sentinel == -4);
  }

  // Test `operator+=`/`operator-=` moving back and forth across a partial final chunk, when the range is
  // not evenly divisible by the chunk size.
  {
    std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> uneven_chunked = vector | std::views::chunk(5);

    auto it = uneven_chunked.begin();
    assert(std::ranges::equal(*it, std::vector{1, 2, 3, 4, 5}));

    it += 2;
    assert(std::ranges::equal(*it, std::vector{11, 12}));

    it -= 1;
    assert(std::ranges::equal(*it, std::vector{6, 7, 8, 9, 10}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
