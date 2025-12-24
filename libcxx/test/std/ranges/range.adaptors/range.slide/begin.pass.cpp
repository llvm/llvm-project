//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   constexpr auto begin() requires (!(__simple_view<V> && __slide_caches_nothing<V>));
//   constexpr auto begin() const requires __slide_caches_nothing<V>;

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>
#include <utility>

constexpr bool test() {
  std::vector<int> vector                                                 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::slide_view<std::ranges::ref_view<std::vector<int>>> slided = vector | std::views::slide(3);
  std::ranges::slide_view<std::ranges::ref_view<const std::vector<int>>> const_slided =
      std::as_const(vector) | std::views::slide(3);

  // Test `slide_view.begin()`
  {
    /*slide_view::__iterator<false>*/ std::forward_iterator auto it = slided.begin();
    assert(std::ranges::equal(*it, std::vector{1, 2, 3}));
    assert(std::ranges::equal(*++it, std::vector{2, 3, 4}));
    assert(std::ranges::equal(*++it, std::vector{3, 4, 5}));
    assert(std::ranges::equal(*++it, std::vector{4, 5, 6}));
    assert(std::ranges::equal(*++it, std::vector{5, 6, 7}));
    assert(std::ranges::equal(*++it, std::vector{6, 7, 8}));
    assert(++it == slided.end());
    /*slide_view::__iterator<true>*/ std::forward_iterator auto const_it = const_slided.begin();
    assert(std::ranges::equal(*const_it, std::vector{1, 2, 3}));
    assert(std::ranges::equal(*++const_it, std::vector{2, 3, 4}));
    assert(std::ranges::equal(*++const_it, std::vector{3, 4, 5}));
    assert(std::ranges::equal(*++const_it, std::vector{4, 5, 6}));
    assert(std::ranges::equal(*++const_it, std::vector{5, 6, 7}));
    assert(std::ranges::equal(*++const_it, std::vector{6, 7, 8}));
    assert(++const_it == const_slided.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
