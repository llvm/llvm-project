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

#include <flat_map>

#include <cassert>

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
  int count = 0;
  std::flat_map<int, int, CountingComp> map({{0, 1}, {0, 2}}, CountingComp(count));
  return map.size() == 1 && map.begin()->first == 0 && count == 2;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
