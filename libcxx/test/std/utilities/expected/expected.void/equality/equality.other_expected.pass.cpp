//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class T2, class E2> requires (is_void_v<T2>)
//   friend constexpr bool operator==(const expected& x, const expected<T2, E2>& y);

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "../../types.h"

struct Foo{};
static_assert(!CanCompare<Foo, Foo>);

static_assert(CanCompare<std::expected<void, int>, std::expected<void, int>>);
static_assert(CanCompare<std::expected<void, int>, std::expected<void, short>>);

#if TEST_STD_VER >= 26
// https://wg21.link/P3379R0
static_assert(!CanCompare<std::expected<void, int>, std::expected<int, int>>);
static_assert(CanCompare<std::expected<void, int>, std::expected<void, int>>);
static_assert(CanCompare<std::expected<void, int>, std::expected<void, int>>);
static_assert(!CanCompare<std::expected<void, NonComparable>, std::expected<void, NonComparable>>);
static_assert(!CanCompare<std::expected<void, int>, std::expected<void, NonComparable>>);
static_assert(!CanCompare<std::expected<void, NonComparable>, std::expected<void, int>>);
#else
// Note this is true because other overloads in expected<non-void> are unconstrained
static_assert(CanCompare<std::expected<void, int>, std::expected<int, int>>);
#endif

constexpr bool test() {
  // x.has_value() && y.has_value()
  {
    const std::expected<void, int> e1;
    const std::expected<void, int> e2;
    assert(e1 == e2);
  }

  // !x.has_value() && y.has_value()
  {
    const std::expected<void, int> e1(std::unexpect, 5);
    const std::expected<void, int> e2;
    assert(e1 != e2);
  }

  // x.has_value() && !y.has_value()
  {
    const std::expected<void, int> e1;
    const std::expected<void, int> e2(std::unexpect, 10);
    const std::expected<void, int> e3(std::unexpect, 5);
    assert(e1 != e2);
    assert(e1 != e3);
  }

  // !x.has_value() && !y.has_value()
  {
    const std::expected<void, int> e1(std::unexpect, 5);
    const std::expected<void, int> e2(std::unexpect, 10);
    const std::expected<void, int> e3(std::unexpect, 5);
    assert(e1 != e2);
    assert(e1 == e3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
