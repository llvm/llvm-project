//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <ranges>
#include <type_traits>

#include "../types.h"

constexpr bool test() {
  // Test the segmented iterator implementation of join_view
  // https://github.com/llvm/llvm-project/issues/158279
  {
    int buffer1[2][1] = {{1}, {2}};
    auto joined       = std::views::join(buffer1);
    assert(std::ranges::find(joined, 1) == std::ranges::begin(joined));
    assert(std::ranges::find(joined, 2) == std::ranges::next(std::ranges::begin(joined)));
    assert(std::ranges::find(joined, 3) == std::ranges::end(joined));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
