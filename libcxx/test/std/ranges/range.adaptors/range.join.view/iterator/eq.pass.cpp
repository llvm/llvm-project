//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr bool operator==(const iterator& x, const iterator& y);

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  std::ranges::join_view jv(buffer);
  auto iter1 = jv.begin();
  auto iter2 = jv.begin();
  assert(iter1 == iter2);
  iter1++;
  assert(iter1 != iter2);
  iter2++;
  assert(iter1 == iter2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
