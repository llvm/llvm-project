//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(const flat_multimap& m);

#include <cassert>
#include <deque>
#include <flat_map>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    using C = test_less<int>;
    KeyContainer<int, test_allocator<int>> ks({1, 1, 3, 5}, test_allocator<int>(6));
    ValueContainer<char, test_allocator<char>> vs({2, 2, 2, 1}, test_allocator<char>(7));
    using M = std::flat_multimap<int, char, C, decltype(ks), decltype(vs)>;
    auto mo = M(ks, vs, C(5));
    auto m  = mo;

    assert(m.key_comp() == C(5));
    assert(m.keys() == ks);
    assert(m.values() == vs);
    assert(m.keys().get_allocator() == test_allocator<int>(6));
    assert(m.values().get_allocator() == test_allocator<char>(7));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(mo.keys() == ks);
    assert(mo.values() == vs);
    assert(mo.keys().get_allocator() == test_allocator<int>(6));
    assert(mo.values().get_allocator() == test_allocator<char>(7));
  }
  {
    using C  = test_less<int>;
    using Ks = KeyContainer<int, other_allocator<int>>;
    using Vs = ValueContainer<char, other_allocator<char>>;
    auto ks  = Ks({1, 1, 3, 5}, other_allocator<int>(6));
    auto vs  = Vs({2, 2, 2, 1}, other_allocator<char>(7));
    using M  = std::flat_multimap<int, char, C, Ks, Vs>;
    auto mo  = M(Ks(ks, other_allocator<int>(6)), Vs(vs, other_allocator<int>(7)), C(5));
    auto m   = mo;

    assert(m.key_comp() == C(5));
    assert(m.keys() == ks);
    assert(m.values() == vs);
    assert(m.keys().get_allocator() == other_allocator<int>(-2));
    assert(m.values().get_allocator() == other_allocator<char>(-2));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(mo.keys() == ks);
    assert(mo.values() == vs);
    assert(mo.keys().get_allocator() == other_allocator<int>(6));
    assert(mo.values().get_allocator() == other_allocator<char>(7));
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
