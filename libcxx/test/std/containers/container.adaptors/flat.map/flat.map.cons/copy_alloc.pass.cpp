//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(const flat_map&, const allocator_type&);

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**)
{
  {
    using C = test_less<int>;
    std::vector<int, test_allocator<int>> ks({1, 3, 5}, test_allocator<int>(6));
    std::vector<char, test_allocator<char>> vs({2, 2, 1}, test_allocator<char>(7));
    using M = std::flat_map<int, char, C, decltype(ks), decltype(vs)>;
    auto mo = M(ks, vs, C(5));
    auto m = M(mo, test_allocator<int>(3));

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
  {
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({1,2,3}, {2,2,1}, C(5), &mr1);
    M m = {mo, &mr2};  // also test the implicitness of this constructor

    assert(m.key_comp() == C(5));
    assert((m.keys() == std::pmr::vector<int>{1,2,3}));
    assert((m.values() == std::pmr::vector<int>{2,2,1}));
    assert(m.keys().get_allocator().resource() == &mr2);
    assert(m.values().get_allocator().resource() == &mr2);

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert((mo.keys() == std::pmr::vector<int>{1,2,3}));
    assert((mo.values() == std::pmr::vector<int>{2,2,1}));
    assert(mo.keys().get_allocator().resource() == &mr1);
    assert(mo.values().get_allocator().resource() == &mr1);
  }
  {
    using M = std::flat_map<int, int, std::less<>, std::pmr::vector<int>, std::pmr::deque<int>>;
    std::pmr::vector<M> vs;
    M m = {{1,2}, {2,2}, {3,1}};
    vs.push_back(m);
    assert(vs[0] == m);
  }
  return 0;
}
