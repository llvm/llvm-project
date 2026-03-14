//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr bool operator==(const iterator& x, const sentinel& y);

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  {
    const std::ranges::iota_view<int, IntComparableWith<int>> io(0, IntComparableWith<int>(10));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }
  {
    std::ranges::iota_view<int, IntComparableWith<int>> io(0, IntComparableWith<int>(10));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }
  {
    const std::ranges::iota_view io(SomeInt(0), IntComparableWith(SomeInt(10)));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }
  {
    std::ranges::iota_view io(SomeInt(0), IntComparableWith(SomeInt(10)));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter != sent);
    assert(iter + 10 == sent);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
