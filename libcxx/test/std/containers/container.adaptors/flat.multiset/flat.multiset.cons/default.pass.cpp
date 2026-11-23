//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_multiset();

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <type_traits>
#include <vector>

#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "test_macros.h"

struct DefaultCtableComp {
  constexpr explicit DefaultCtableComp() { default_constructed_ = true; }
  constexpr bool operator()(int, int) const { return false; }
  bool default_constructed_ = false;
};

struct ThrowingCtorComp {
  constexpr ThrowingCtorComp() noexcept(false) {}
  constexpr bool operator()(const auto&, const auto&) const { return false; }
};

template <template <class...> class KeyContainer>
constexpr void test() {
  {
    std::flat_multiset<int, std::less<int>, KeyContainer<int>> m;
    assert(m.empty());
  }
  {
    // explicit(false)
    std::flat_multiset<int, std::less<int>, KeyContainer<int>> m = {};
    assert(m.empty());
  }
  {
    std::flat_multiset<int, DefaultCtableComp, KeyContainer<int, min_allocator<int>>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp().default_constructed_);
  }
  {
    using A1 = explicit_allocator<int>;
    {
      std::flat_multiset<int, DefaultCtableComp, KeyContainer<int, A1>> m;
      assert(m.empty());
      assert(m.key_comp().default_constructed_);
    }
    {
      A1 a1;
      std::flat_multiset<int, DefaultCtableComp, std::vector<int, A1>> m(a1);
      assert(m.empty());
      assert(m.key_comp().default_constructed_);
    }
  }
#if defined(_LIBCPP_VERSION)
  {
    using C = std::flat_multiset<MoveOnly, std::less<MoveOnly>>;
    static_assert(std::is_nothrow_default_constructible_v<C>);
    C c;
  }
  {
    using C = std::flat_multiset<MoveOnly, std::less<MoveOnly>, KeyContainer<MoveOnly, test_allocator<MoveOnly>>>;
    static_assert(std::is_nothrow_default_constructible_v<C>);
    C c;
  }
#endif // _LIBCPP_VERSION
  {
    using C = std::flat_multiset<MoveOnly, std::less<MoveOnly>, KeyContainer<MoveOnly, other_allocator<MoveOnly>>>;
    static_assert(!std::is_nothrow_default_constructible_v<C>);
    C c;
  }
  {
    using C = std::flat_multiset<MoveOnly, ThrowingCtorComp, KeyContainer<MoveOnly>>;
    static_assert(!std::is_nothrow_default_constructible_v<C>);
    C c;
  }
}

constexpr bool test() {
  test<std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque>();
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
