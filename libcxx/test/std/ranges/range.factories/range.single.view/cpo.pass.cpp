//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::single

#include <ranges>

#include <cassert>
#include <concepts>
#include <utility>
#include "MoveOnly.h"

// Can't invoke without arguments.
static_assert(!std::is_invocable_v<decltype((std::views::single))>);
#if _LIBCPP_STD_VER >= 23
// Can invoke with a move-only type.
static_assert(std::is_invocable_v<decltype((std::views::single)), MoveOnly>);
#endif
constexpr bool test() {
  // Lvalue.
  {
    int x = 42;
    std::same_as<std::ranges::single_view<int>> decltype(auto) v = std::views::single(x);
    assert(v.size() == 1);
    assert(v.front() == x);
  }

  // Prvalue.
  {
    std::same_as<std::ranges::single_view<int>> decltype(auto) v = std::views::single(42);
    assert(v.size() == 1);
    assert(v.front() == 42);
  }

  // Const lvalue.
  {
    const int x = 42;
    std::same_as<std::ranges::single_view<int>> decltype(auto) v = std::views::single(x);
    assert(v.size() == 1);
    assert(v.front() == x);
  }

  // Xvalue.
  {
    int x = 42;
    std::same_as<std::ranges::single_view<int>> decltype(auto) v = std::views::single(std::move(x));
    assert(v.size() == 1);
    assert(v.front() == x);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
