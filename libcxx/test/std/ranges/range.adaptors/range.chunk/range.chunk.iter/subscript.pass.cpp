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
#include <iterator>
#include <ranges>
#include <vector>

#include "test_range.h"

constexpr bool test() {
  std::vector<int> vector                                                  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(3);

  // Test `constexpr value_type iterator::operator[](difference_type n) const`
  {
    assert(std::ranges::equal(chunked.begin()[1], std::vector{4, 5, 6}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
