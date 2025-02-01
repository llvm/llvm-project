//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set& operator=(flat_set&&);

#include <algorithm>
#include <deque>
#include <flat_set>
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
    using M  = std::flat_set<int, C, std::vector<int, A1>>;
    M mo     = M({1, 2, 3}, C(5), A1(7));
    M m      = M({}, C(3), A1(7));
    m        = std::move(mo);
    assert((m == M{1, 2, 3}));
    assert(m.key_comp() == C(5));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(mo.empty());
  }
  {
    using C  = test_less<int>;
    using A1 = other_allocator<int>;
    using M  = std::flat_set<int, C, std::deque<int, A1>>;
    M mo     = M({4, 5}, C(5), A1(7));
    M m      = M({1, 2, 3, 4}, C(3), A1(7));
    m        = std::move(mo);
    assert((m == M{4, 5}));
    assert(m.key_comp() == C(5));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(mo.empty());
  }
  {
    using A = min_allocator<int>;
    using M = std::flat_set<int, std::greater<int>, std::vector<int, A>>;
    M mo    = M({5, 4, 3}, A());
    M m     = M({4, 3, 2, 1}, A());
    m       = std::move(mo);
    assert((m == M{5, 4, 3}));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator() == A());
    assert(mo.empty());
  }

  return 0;
}
