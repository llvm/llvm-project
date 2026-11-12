//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr decltype(auto) operator*() const;

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    std::ranges::join_view jv(buffer);
    auto iter = jv.begin();
    for (int i = 1; i < 17; ++i) {
      assert(*iter++ == i);
    }
  }
  {
    std::ranges::join_view jv(buffer);
    auto iter = std::next(jv.begin(), 15);
    assert(*iter++ == 16);
    assert(iter == jv.end());
  }
  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv = std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    for (int i = 1; i < 17; ++i) {
      assert(*iter == i);
      ++iter;
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
