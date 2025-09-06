//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <array>
#include <cassert>
#include <iterator>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  // Test two iterators
  {
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    auto it1 = view.begin();
    it1++;
    it1++;
    auto it2 = view.begin();
    auto res = it1 - it2;
    assert(res == 2);
  }

  // Test one iterator and one sentinel
  {
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    auto it1 = view.begin();
    auto res = std::default_sentinel_t{} - it1;
    assert(res == 4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
