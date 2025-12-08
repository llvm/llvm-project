//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::chunk

#include <ranges>

#include <array>
#include <cassert>
#include <span>

#include "test_range.h"

constexpr bool test() {
  std::array array = {1, 1, 1, 2, 2, 2, 3, 3};

  // Test `chunk_view.base()`
  {
    auto view = array | std::views::chunk(3);
    auto base = view.begin().base();
    assert(base == array.begin());
    base = view.end().base();
    assert(base == array.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}