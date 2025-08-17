//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class T2> friend constexpr bool operator==(const expected& x, const T2& v);

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26
// https://wg21.link/P3379R0
static_assert(HasOperatorEqual<std::expected<int, int>, int>);
static_assert(HasOperatorEqual<std::expected<int, int>, EqualityComparable>);
static_assert(!HasOperatorEqual<std::expected<int, int>, NonComparable>);
#endif

constexpr bool test() {
  // x.has_value()
  {
    const std::expected<EqualityComparable, int> e1(std::in_place, 5);
    int i2 = 10;
    int i3 = 5;
    assert(e1 != i2);
    assert(e1 == i3);
  }

  // !x.has_value()
  {
    const std::expected<EqualityComparable, int> e1(std::unexpect, 5);
    int i2 = 10;
    int i3 = 5;
    assert(e1 != i2);
    assert(e1 != i3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
