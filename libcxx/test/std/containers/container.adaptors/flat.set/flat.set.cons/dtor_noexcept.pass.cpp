//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// ~flat_set();

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

struct ThrowingDtorComp {
  constexpr bool operator()(const auto&, const auto&) const;
  constexpr ~ThrowingDtorComp() noexcept(false) {}
};

template <template <class...> class KeyContainer>
constexpr void test() {
  {
    using C = std::flat_set<MoveOnly, std::less<MoveOnly>, KeyContainer<MoveOnly>>;
    static_assert(std::is_nothrow_destructible_v<C>);
    C c;
  }
  {
    using V = KeyContainer<MoveOnly, test_allocator<MoveOnly>>;
    using C = std::flat_set<MoveOnly, std::less<MoveOnly>, V>;
    static_assert(std::is_nothrow_destructible_v<C>);
    C c;
  }
  {
    using V = KeyContainer<MoveOnly, other_allocator<MoveOnly>>;
    using C = std::flat_set<MoveOnly, std::greater<MoveOnly>, V>;
    static_assert(std::is_nothrow_destructible_v<C>);
    C c;
  }
#if defined(_LIBCPP_VERSION)
  {
    using C = std::flat_set<MoveOnly, ThrowingDtorComp, KeyContainer<MoveOnly>>;
    static_assert(!std::is_nothrow_destructible_v<C>);
    C c;
  }
#endif // _LIBCPP_VERSION
}

constexpr bool test() {
  test<std::vector>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
