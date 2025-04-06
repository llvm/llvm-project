//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(const flat_map& m);

#include <cassert>
#include <flat_map>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    using C = test_less<int>;
    std::vector<int, test_allocator<int>> ks({1, 3, 5}, test_allocator<int>(6));
    std::vector<char, test_allocator<char>> vs({2, 2, 1}, test_allocator<char>(7));
    using M = std::flat_map<int, char, C, decltype(ks), decltype(vs)>;
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
    using Ks = std::vector<int, other_allocator<int>>;
    using Vs = std::vector<char, other_allocator<char>>;
    auto ks  = Ks({1, 3, 5}, other_allocator<int>(6));
    auto vs  = Vs({2, 2, 1}, other_allocator<char>(7));
    using M  = std::flat_map<int, char, C, Ks, Vs>;
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

  return 0;
}
