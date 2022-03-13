//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr iterator& operator--();
// constexpr iterator operator--(int);

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    // outer == ranges::end
    std::ranges::join_view jv(buffer);
    auto iter = std::next(jv.begin(), 16);
    for (int i = 16; i != 0; --i) {
      assert(*--iter == i);
    }
  }
  {
    // outer == ranges::end
    std::ranges::join_view jv(buffer);
    auto iter = std::next(jv.begin(), 13);
    for (int i = 13; i != 0; --i) {
      assert(*--iter == i);
    }
  }
  {
    // outer != ranges::end
    std::ranges::join_view jv(buffer);
    auto iter = std::next(jv.begin(), 12);
    for (int i = 12; i != 0; --i) {
      assert(*--iter == i);
    }
  }
  {
    // outer != ranges::end
    std::ranges::join_view jv(buffer);
    auto iter = std::next(jv.begin());
    for (int i = 1; i != 0; --i) {
      assert(*--iter == i);
    }
  }
  {
    int small[2][1] = {{1}, {2}};
    std::ranges::join_view jv(small);
    auto iter = std::next(jv.begin(), 2);
    for (int i = 2; i != 0; --i) {
      assert(*--iter == i);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
