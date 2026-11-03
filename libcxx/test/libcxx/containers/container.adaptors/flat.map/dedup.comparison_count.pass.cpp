//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// libc++ extension: sorted deduplication uses one-direction key comparison.

#include <algorithm>
#include <cassert>
#include <flat_map>
#include <utility>
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
    std::flat_map<int, int, CountingComp> map({{0, 1}, {0, 2}}, CountingComp(count));
    if (map.size() != 1 || map.begin()->first != 0 || count != 2)
      return false;
  }
  {
    // Interleaved duplicate runs exercise the element-moving part of the deduplication loop.
    int count = 0;
    std::flat_map<int, int, CountingComp> map({{1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 1}, {4, 1}, {4, 2}},
                                              CountingComp(count));
    // Which mapped value survives deduplication is unspecified, so only check the keys.
    int expected_keys[] = {1, 2, 3, 4};
    if (!std::ranges::equal(map.keys(), expected_keys))
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
