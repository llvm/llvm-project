//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Some basic examples of how adjacent_transform_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <cstddef>
#include <functional>
#include <ranges>

#include <cassert>
#include <vector>

constexpr bool test() {
  std::vector v        = {1, 2, 3, 4};
  std::vector expected = {2, 6, 12};

  {
    auto expected_index = 0;
    for (auto x : v | std::views::adjacent_transform<2>(std::multiplies())) {
      assert(x == expected[expected_index++]);
    }
  }
  {
    auto expected_index = 0;
    for (auto x : v | std::views::pairwise_transform(std::multiplies())) {
      assert(x == expected[expected_index++]);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
