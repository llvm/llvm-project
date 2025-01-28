//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map& operator=(flat_map&&);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<char>;
    using M  = std::flat_map<int, char, C, std::vector<int, A1>, std::vector<char, A2>>;
    M mo     = M({{1, 1}, {2, 3}, {3, 2}}, C(5), A1(7));
    M m      = M({}, C(3), A1(7));
    m        = std::move(mo);
    assert((m == M{{1, 1}, {2, 3}, {3, 2}}));
    assert(m.key_comp() == C(5));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(vs.get_allocator() == A2(7));
    assert(mo.empty());
  }
  {
    using C  = test_less<int>;
    using A1 = other_allocator<int>;
    using A2 = other_allocator<char>;
    using M  = std::flat_map<int, char, C, std::deque<int, A1>, std::deque<char, A2>>;
    M mo     = M({{4, 5}, {5, 4}}, C(5), A1(7));
    M m      = M({{1, 1}, {2, 2}, {3, 3}, {4, 4}}, C(3), A1(7));
    m        = std::move(mo);
    assert((m == M{{4, 5}, {5, 4}}));
    assert(m.key_comp() == C(5));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(vs.get_allocator() == A2(7));
    assert(mo.empty());
  }
  {
    using A = min_allocator<int>;
    using M = std::flat_map<int, int, std::greater<int>, std::vector<int, A>, std::vector<int, A>>;
    M mo    = M({{5, 1}, {4, 2}, {3, 3}}, A());
    M m     = M({{4, 4}, {3, 3}, {2, 2}, {1, 1}}, A());
    m       = std::move(mo);
    assert((m == M{{5, 1}, {4, 2}, {3, 3}}));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A());
    assert(vs.get_allocator() == A());
    assert(mo.empty());
  }

  return 0;
}
