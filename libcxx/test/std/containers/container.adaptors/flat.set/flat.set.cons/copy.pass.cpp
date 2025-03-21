//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set(const flat_set& m);

#include <algorithm>
#include <cassert>
#include <flat_set>
#include <vector>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    using C = test_less<int>;
    std::vector<int, test_allocator<int>> ks({1, 3, 5}, test_allocator<int>(6));
    using M = std::flat_set<int, C, decltype(ks)>;
    auto mo = M(ks, C(5));
    auto m  = mo;

    assert(m.key_comp() == C(5));
    assert(std::ranges::equal(m, ks));
    auto keys = std::move(m).extract();
    assert(keys.get_allocator() == test_allocator<int>(6));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(std::ranges::equal(mo, ks));
    auto keys2 = std::move(mo).extract();
    assert(keys2.get_allocator() == test_allocator<int>(6));
  }
  {
    using C  = test_less<int>;
    using Ks = std::vector<int, other_allocator<int>>;
    auto ks  = Ks({1, 3, 5}, other_allocator<int>(6));
    using M  = std::flat_set<int, C, Ks>;
    auto mo  = M(Ks(ks, other_allocator<int>(6)), C(5));
    auto m   = mo;

    assert(m.key_comp() == C(5));
    assert(std::ranges::equal(m, ks));
    auto keys = std::move(m).extract();
    assert(keys.get_allocator() == other_allocator<int>(-2));

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert(std::ranges::equal(mo, ks));
    auto keys2 = std::move(mo).extract();
    assert(keys2.get_allocator() == other_allocator<int>(6));
  }

  return 0;
}
