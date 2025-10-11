//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class Allocator>
//   explicit flat_multiset(const Allocator& a);

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../../test_compare.h"

template <template <class...> class KeyContainer>
constexpr void test() {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<container_type, Alloc> is true

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using V1 = KeyContainer<int, A1>;
    using V2 = KeyContainer<int, A2>;
    using M1 = std::flat_multiset<int, C, V1>;
    using M2 = std::flat_multiset<int, C, V2>;
    static_assert(std::is_constructible_v<M1, const A1&>);
    static_assert(std::is_constructible_v<M2, const A2&>);
    static_assert(!std::is_constructible_v<M1, const A2&>);
    static_assert(!std::is_constructible_v<M2, const A1&>);
  }
  {
    using A = test_allocator<short>;
    using M = std::flat_multiset<int, std::less<int>, KeyContainer<int, test_allocator<int>>>;
    M m(A(0, 5));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(std::move(m).extract().get_allocator().get_id() == 5);
  }
  {
    // explicit
    using M = std::flat_multiset<int, std::less<int>, KeyContainer<int, test_allocator<int>>>;

    static_assert(std::is_constructible_v<M, test_allocator<int>>);
    static_assert(!std::is_convertible_v<test_allocator<int>, M>);
  }
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
