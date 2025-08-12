//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// explicit flat_multimap(const key_compare& comp);
// template <class Alloc>
//   flat_multimap(const key_compare& comp, const Alloc& a);

#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "MinSequenceContainer.h"
#include "min_allocator.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

// explicit flat_multimap(const key_compare& comp);
template <class KeyContainer, class ValueContainer>
constexpr void test_compare() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  {
    // The one-argument ctor is explicit.
    using C = test_less<Key>;
    static_assert(std::is_constructible_v<std::flat_multimap<Key, Value, C>, C>);
    static_assert(!std::is_convertible_v<C, std::flat_multimap<Key, Value, C>>);

    static_assert(std::is_constructible_v<std::flat_multimap<Key, Value>, std::less<Key>>);
    static_assert(!std::is_convertible_v<std::less<Key>, std::flat_multimap<Key, Value>>);
  }
  {
    using C = test_less<Key>;
    auto m  = std::flat_multimap<Key, Value, C>(C(3));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(3));
  }
}

// template <class Alloc>
//   flat_multimap(const key_compare& comp, const Alloc& a);
template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test_compare_alloc() {
  {
    // If an allocator is given, it must be usable by both containers.
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<>, KeyContainer<int>, ValueContainer<int, A>>;
    static_assert(std::is_constructible_v<M, std::less<>>);
    static_assert(!std::is_constructible_v<M, std::less<>, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::less<>, A>);
  }
  {
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    auto m   = std::flat_multimap<int, short, C, KeyContainer<int, A1>, ValueContainer<short, A2>>(C(4), A1(5));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // explicit(false)
    using C                                                                               = test_less<int>;
    using A1                                                                              = test_allocator<int>;
    using A2                                                                              = test_allocator<short>;
    std::flat_multimap<int, short, C, KeyContainer<int, A1>, ValueContainer<short, A2>> m = {C(4), A1(5)};
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
}

constexpr bool test() {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<key_container_type, Alloc> is true
    // and uses_allocator_v<mapped_container_type, Alloc> is true.

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using M1 = std::flat_multimap<int, int, C, std::vector<int, A1>, std::vector<int, A1>>;
    using M2 = std::flat_multimap<int, int, C, std::vector<int, A1>, std::vector<int, A2>>;
    using M3 = std::flat_multimap<int, int, C, std::vector<int, A2>, std::vector<int, A1>>;
    static_assert(std::is_constructible_v<M1, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const C&, const A2&>);
  }

  test_compare<std::vector<int>, std::vector<int>>();
  test_compare<std::vector<int>, std::vector<double>>();
  test_compare<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test_compare<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();
  test_compare<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();

  test_compare_alloc<std::vector, std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test_compare<std::deque<int>, std::vector<double>>();
    test_compare_alloc<std::deque, std::deque>();
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
