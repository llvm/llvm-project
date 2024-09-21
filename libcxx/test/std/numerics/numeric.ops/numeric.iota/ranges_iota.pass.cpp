//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <numeric>

// iota_result<O, T> iota(O first, S last, T value);

#include <algorithm>
#include <cassert>
#include <numeric>
#include <ranges>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

constexpr bool test() {
  { // empty range
    std::vector<int> vec;
    constexpr int value = 42;
    for (int i = 0; i < 2; ++i) {
      const auto [last, final_value] =
          (i == 0) ? std::ranges::iota(vec.begin(), vec.end(), value)
                   : std::ranges::iota<decltype(vec.begin())>(vec, value);
      assert(vec.empty());
      assert(last == vec.end());
      assert(final_value == value);
    }
  }

  { // non-empty range
    constexpr int size = 3;
    std::vector<int> vec(size);
    constexpr int value = 42;
    for (int i = 0; i < 2; ++i) {
      const auto [last, final_value] =
          (i == 0) ? std::ranges::iota(vec.begin(), vec.end(), value)
                   : std::ranges::iota<decltype(vec.begin())>(vec, value);
      assert(std::ranges::equal(vec, std::vector{value, value + 1, value + 2}));
      assert(last == vec.end());
      assert(final_value == value + size);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
}
