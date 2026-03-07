//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   constexpr iterator& operator--()
//     requires bidirectional_range<Base>;
//   constexpr iterator operator--(int)
//     requires bidirectional_range<Base>;
//   constexpr iterator& operator-=(difference_type)
//     requires random_access_range<Base>;

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

constexpr bool test() {
  std::vector<int> vector                                                 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::slide_view<std::ranges::ref_view<std::vector<int>>> slided = vector | std::views::slide(2);

  // Test `constexpr iterator& operator--();`
  {
    /*slide_view::__iterator<false>*/ std::bidirectional_iterator auto it = slided.end();
    assert(std::ranges::equal(*--it, std::vector{7, 8}));
  }

  // Test `constexpr iterator operator--(int)`
  {
    /*slide_view::__iterator<false>*/ std::bidirectional_iterator auto it = slided.end();
    it--;
    assert(std::ranges::equal(*it, std::vector{7, 8}));
  }

  // Test `constexpr iterator& operator-=(difference_type)`
  {
    /*slide_view::__iterator<false>*/ std::random_access_iterator auto it = slided.end();
    it -= 3;
    assert(std::ranges::equal(*it, std::vector{5, 6}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
