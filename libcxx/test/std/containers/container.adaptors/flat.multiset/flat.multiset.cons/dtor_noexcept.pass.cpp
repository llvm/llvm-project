//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// ~flat_multiset();

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

struct ThrowingDtorComp {
  bool operator()(const auto&, const auto&) const;
  ~ThrowingDtorComp() noexcept(false) {}
};

void test() {
  {
    using C = std::flat_multiset<MoveOnly, MoveOnly>;
    static_assert(std::is_nothrow_destructible_v<C>);
    C c;
  }
  {
    using V = std::vector<MoveOnly, test_allocator<MoveOnly>>;
    using C = std::flat_multiset<MoveOnly, std::less<MoveOnly>, V>;
    static_assert(std::is_nothrow_destructible_v<C>);
    C c;
  }
  {
    using V = std::deque<MoveOnly, other_allocator<MoveOnly>>;
    using C = std::flat_multiset<MoveOnly, std::greater<MoveOnly>, V>;
    static_assert(std::is_nothrow_destructible_v<C>);
    C c;
  }
#if defined(_LIBCPP_VERSION)
  {
    using C = std::flat_multiset<MoveOnly, ThrowingDtorComp>;
    static_assert(!std::is_nothrow_destructible_v<C>);
    C c;
  }
#endif // _LIBCPP_VERSION
}

int main(int, char**) {
  test();

  return 0;
}
