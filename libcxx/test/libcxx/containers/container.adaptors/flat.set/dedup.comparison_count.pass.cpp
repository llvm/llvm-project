//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// libc++ extension: sorted deduplication uses one-direction key comparison.

#include <algorithm>
#include <cassert>
#include <flat_set>
#include <vector>

#include "test_macros.h"

struct CountingComp {
  constexpr explicit CountingComp(int& count) : count_(count) {}

  constexpr bool operator()(int x, int y) const {
    ++count_;
    return x < y;
  }

  int& count_;
};

constexpr bool test() {
  {
    int count = 0;
    std::flat_set<int, CountingComp> set({0, 0}, CountingComp(count));
    if (set.size() != 1 || count != 2)
      return false;
  }
  {
    // Interleaved duplicate runs exercise the element-moving part of the deduplication loop.
    int count = 0;
    std::flat_set<int, CountingComp> set({1, 1, 2, 2, 2, 3, 4, 4}, CountingComp(count));
    int expected[] = {1, 2, 3, 4};
    if (!std::ranges::equal(set, expected))
      return false;
    (void)count;
  }
  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
