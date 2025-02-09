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
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    // test_allocator is not propagated
    using C = test_less<int>;
    std::vector<int, test_allocator<int>> ks({1, 1, 3, 3, 5}, test_allocator<int>(6));
    std::vector<char, test_allocator<char>> vs({1, 2, 3, 4, 5}, test_allocator<char>(7));
    using M = std::flat_multimap<int, char, C, decltype(ks), decltype(vs)>;
    auto mo = M(ks, vs, C(5));
    auto m  = M({{3, 3}, {4, 4}, {5, 5}}, C(3), test_allocator<int>(2));
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
    using Ks = std::vector<int, other_allocator<int>>;
    using Vs = std::vector<char, other_allocator<char>>;
    auto ks  = Ks({1, 1, 3, 3, 5}, other_allocator<int>(6));
    auto vs  = Vs({2, 1, 3, 2, 1}, other_allocator<char>(7));
    using M  = std::flat_multimap<int, char, C, Ks, Vs>;
    auto mo  = M(Ks(ks, other_allocator<int>(6)), Vs(vs, other_allocator<int>(7)), C(5));
    auto m   = M({{3, 3}, {4, 4}, {5, 5}}, C(3), other_allocator<int>(2));
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
  {
    // self-assignment
    using M = std::flat_multimap<int, int>;
    M m     = {{1, 1}, {3, 4}};
    m       = static_cast<const M&>(m);
    assert((m == M{{1, 1}, {3, 4}}));
  }
  return 0;
}
