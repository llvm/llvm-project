//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap& operator=(const flat_multimap& m);

#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    // test_allocator is not propagated
    using C = test_less<int>;
    KeyContainer<int, test_allocator<int>> ks({1, 1, 3, 5}, test_allocator<int>(6));
    ValueContainer<char, test_allocator<char>> vs({2, 2, 2, 1}, test_allocator<char>(7));
    using M = std::flat_multimap<int, char, C, decltype(ks), decltype(vs)>;
    auto mo = M(ks, vs, C(5));
    auto m  = M({{3, 3}, {4, 4}, {5, 5}, {5, 5}}, C(3), test_allocator<int>(2));
    m       = mo;

    assert(m.key_comp() == C(5));
    assert(m.keys() == ks);
    assert(m.values() == vs);
    assert(m.keys().get_allocator() == test_allocator<int>(2));
    assert(m.values().get_allocator() == test_allocator<char>(2));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(mo.keys() == ks);
    assert(mo.values() == vs);
    assert(mo.keys().get_allocator() == test_allocator<int>(6));
    assert(mo.values().get_allocator() == test_allocator<char>(7));
  }
  {
    // other_allocator is propagated
    using C  = test_less<int>;
    using Ks = KeyContainer<int, other_allocator<int>>;
    using Vs = ValueContainer<char, other_allocator<char>>;
    auto ks  = Ks({1, 1, 3, 5}, other_allocator<int>(6));
    auto vs  = Vs({2, 2, 2, 1}, other_allocator<char>(7));
    using M  = std::flat_multimap<int, char, C, Ks, Vs>;
    auto mo  = M(Ks(ks, other_allocator<int>(6)), Vs(vs, other_allocator<int>(7)), C(5));
    auto m   = M({{3, 3}, {4, 4}, {5, 5}, {5, 5}}, C(3), other_allocator<int>(2));
    m        = mo;

    assert(m.key_comp() == C(5));
    assert(m.keys() == ks);
    assert(m.values() == vs);
    assert(m.keys().get_allocator() == other_allocator<int>(6));
    assert(m.values().get_allocator() == other_allocator<char>(7));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(mo.keys() == ks);
    assert(mo.values() == vs);
    assert(mo.keys().get_allocator() == other_allocator<int>(6));
    assert(mo.values().get_allocator() == other_allocator<char>(7));
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // comparator is copied and invariant is preserved
    using M = std::flat_multimap<int, int, std::function<bool(int, int)>>;
    M mo    = M({{1, 2}, {3, 4}}, std::less<int>());
    M m     = M({{1, 2}, {3, 4}}, std::greater<int>());
    assert(m.key_comp()(2, 1) == true);
    assert(m != mo);
    m = mo;
    assert(m.key_comp()(2, 1) == false);
    assert(m == mo);
  }
  {
    // self-assignment
    using M = std::flat_multimap<int, int>;
    M m     = {{1, 2}, {3, 4}};
    m       = static_cast<const M&>(m);
    assert((m == M{{1, 2}, {3, 4}}));
  }
}

constexpr bool test() {
  test<std::vector, std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque, std::deque>();
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
