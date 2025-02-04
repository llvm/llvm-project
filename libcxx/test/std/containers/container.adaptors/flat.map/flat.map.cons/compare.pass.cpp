//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// explicit flat_map(const key_compare& comp);
// template <class Alloc>
//   flat_map(const key_compare& comp, const Alloc& a);

#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<key_container_type, Alloc> is true
    // and uses_allocator_v<mapped_container_type, Alloc> is true.

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using M1 = std::flat_map<int, int, C, std::vector<int, A1>, std::vector<int, A1>>;
    using M2 = std::flat_map<int, int, C, std::vector<int, A1>, std::vector<int, A2>>;
    using M3 = std::flat_map<int, int, C, std::vector<int, A2>, std::vector<int, A1>>;
    static_assert(std::is_constructible_v<M1, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const C&, const A2&>);
  }
  {
    using C = test_less<int>;
    auto m  = std::flat_map<int, char*, C>(C(3));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(3));
  }
  {
    // The one-argument ctor is explicit.
    using C = test_less<int>;
    static_assert(std::is_constructible_v<std::flat_map<int, char*, C>, C>);
    static_assert(!std::is_convertible_v<C, std::flat_map<int, char*, C>>);

    static_assert(std::is_constructible_v<std::flat_map<int, char*>, std::less<int>>);
    static_assert(!std::is_convertible_v<std::less<int>, std::flat_map<int, char*>>);
  }
  {
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    auto m   = std::flat_map<int, short, C, std::vector<int, A1>, std::vector<short, A2>>(C(4), A1(5));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // explicit(false)
    using C                                                                    = test_less<int>;
    using A1                                                                   = test_allocator<int>;
    using A2                                                                   = test_allocator<short>;
    std::flat_map<int, short, C, std::deque<int, A1>, std::deque<short, A2>> m = {C(4), A1(5)};
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // If an allocator is given, it must be usable by both containers.
    using A = test_allocator<int>;
    using M = std::flat_map<int, int, std::less<>, std::vector<int>, std::vector<int, A>>;
    static_assert(std::is_constructible_v<M, std::less<>>);
    static_assert(!std::is_constructible_v<M, std::less<>, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::less<>, A>);
  }

  return 0;
}
