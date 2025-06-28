//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>
#include <vector>

#include <cassert>
#include "test_iterators.h"

constexpr void general_tests() {
  std::vector<int> v1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> v2 = {1, 2, 3, 4, 5, 6, 7, 8};
  // Check the return type of `.begin()`
  {
    std::ranges::concat_view view(v1, v2);
    using ConcatIterator = std::ranges::iterator_t<decltype(view)>;
    ASSERT_SAME_TYPE(ConcatIterator, decltype(view.begin()));
  }
}

constexpr bool test() {
  general_tests();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
