//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set(flat_set&&);

#include <algorithm>
#include <deque>
#include <flat_set>
#include <functional>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    using C = test_less<int>;
    using A = test_allocator<int>;
    using M = std::flat_set<int, C, std::deque<int, A>>;
    M mo    = M({1, 2, 3}, C(5), A(7));
    M m     = std::move(mo);
    assert((m == M{1, 2, 3}));
    assert(m.key_comp() == C(5));
    assert(std::move(m).extract().get_allocator() == A(7));

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator().get_id() == test_alloc_base::moved_value);
  }
  {
    using C = test_less<int>;
    using A = min_allocator<int>;
    using M = std::flat_set<int, C, std::vector<int, A>>;
    M mo    = M({1, 2, 3}, C(5), A());
    M m     = std::move(mo);
    assert((m == M{1, 2, 3}));
    assert(m.key_comp() == C(5));
    assert(std::move(m).extract().get_allocator() == A());

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator() == A());
  }
  {
    // A moved-from flat_set maintains its class invariant in the presence of moved-from comparators.
    using M = std::flat_set<int, std::function<bool(int, int)>>;
    M mo    = M({1, 2, 3}, std::less<int>());
    M m     = std::move(mo);
    assert(m.size() == 3);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.key_comp()(1, 2) == true);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    LIBCPP_ASSERT(m.key_comp()(1, 2) == true);
    LIBCPP_ASSERT(mo.empty());
    mo.insert({1, 2, 3}); // insert has no preconditions
    assert(m == mo);
  }
  {
    // moved-from object maintains invariant if the underlying container does not clear after move
    using M = std::flat_set<int, std::less<>, CopyOnlyVector<int>>;
    M m1    = M({1, 2, 3});
    M m2    = std::move(m1);
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.size() == 0);
  }
  return 0;
}
