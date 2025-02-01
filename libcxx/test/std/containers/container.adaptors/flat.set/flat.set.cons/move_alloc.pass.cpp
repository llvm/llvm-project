//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set(flat_set&&, const allocator_type&);

#include <algorithm>
#include <deque>
#include <flat_set>
#include <functional>
#include <ranges>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<container_type, Alloc> is true.

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using V1 = std::vector<int, A1>;
    using V2 = std::vector<int, A2>;
    using M1 = std::flat_set<int, C, V1>;
    using M2 = std::flat_set<int, C, V2>;
    static_assert(std::is_constructible_v<M1, M1&&, const A1&>);
    static_assert(std::is_constructible_v<M2, M2&&, const A2&>);
    static_assert(!std::is_constructible_v<M1, M1&&, const A2&>);
    static_assert(!std::is_constructible_v<M2, M2&&, const A1&>);
  }
  {
    int expected[] = {1, 2, 3};
    using C        = test_less<int>;
    using A        = test_allocator<int>;
    using M        = std::flat_set<int, C, std::deque<int, A>>;
    auto mo        = M(expected, expected + 3, C(5), A(7));
    auto m         = M(std::move(mo), A(3));

    assert(m.key_comp() == C(5));
    assert(m.size() == 3);
    auto keys = std::move(m).extract();
    assert(keys.get_allocator() == A(3));
    assert(std::ranges::equal(keys, expected ));

    // The original flat_set is moved-from.
    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator() == A(7));
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_set<int, std::less<>,  CopyOnlyVector<int>>;
    M m1    = M({1, 2, 3});
    M m2(std::move(m1), std::allocator<int>{});
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
  }

  return 0;
}
