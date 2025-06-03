//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(const flat_multimap&, const allocator_type&);

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
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
    using V1 = std::vector<int, A1>;
    using V2 = std::vector<int, A2>;
    using M1 = std::flat_multimap<int, int, C, V1, V1>;
    using M2 = std::flat_multimap<int, int, C, V1, V2>;
    using M3 = std::flat_multimap<int, int, C, V2, V1>;
    static_assert(std::is_constructible_v<M1, const M1&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const M1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const M2&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const M3&, const A2&>);
  }
  {
    using C = test_less<int>;
    std::vector<int, test_allocator<int>> ks({1, 3, 3, 5, 5}, test_allocator<int>(6));
    std::vector<char, test_allocator<char>> vs({2, 2, 1, 1, 1}, test_allocator<char>(7));
    using M = std::flat_multimap<int, char, C, decltype(ks), decltype(vs)>;
    auto mo = M(ks, vs, C(5));
    auto m  = M(mo, test_allocator<int>(3));

    assert(m.key_comp() == C(5));
    assert(m.keys() == ks);
    assert(m.values() == vs);
    assert(m.keys().get_allocator() == test_allocator<int>(3));
    assert(m.values().get_allocator() == test_allocator<char>(3));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(mo.keys() == ks);
    assert(mo.values() == vs);
    assert(mo.keys().get_allocator() == test_allocator<int>(6));
    assert(mo.values().get_allocator() == test_allocator<char>(7));
  }

  return 0;
}
