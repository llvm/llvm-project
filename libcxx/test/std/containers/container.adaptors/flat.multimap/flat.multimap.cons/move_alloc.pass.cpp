//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(flat_multimap&&, const allocator_type&);

#include <algorithm>
#include <deque>
#include <flat_map>
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
    static_assert(std::is_constructible_v<M1, M1&&, const A1&>);
    static_assert(!std::is_constructible_v<M1, M1&&, const A2&>);
    static_assert(!std::is_constructible_v<M2, M2&&, const A2&>);
    static_assert(!std::is_constructible_v<M3, M3&&, const A2&>);
  }
  {
    std::pair<int, int> expected[] = {{1, 1}, {1, 2}, {2, 3}, {2, 2}, {3, 1}};
    using C                        = test_less<int>;
    using A                        = test_allocator<int>;
    using M                        = std::flat_multimap<int, int, C, std::vector<int, A>, std::deque<int, A>>;
    auto mo                        = M(expected, expected + 5, C(5), A(7));
    auto m                         = M(std::move(mo), A(3));

    assert(m.key_comp() == C(5));
    assert(m.size() == 5);
    auto [keys, values] = std::move(m).extract();
    assert(keys.get_allocator() == A(3));
    assert(values.get_allocator() == A(3));
    assert(std::ranges::equal(keys, expected | std::views::elements<0>));
    assert(std::ranges::equal(values, expected | std::views::elements<1>));

    // The original flat_multimap is moved-from.
    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(mo.keys().get_allocator() == A(7));
    assert(mo.values().get_allocator() == A(7));
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_multimap<int, int, std::less<>, std::vector<int>, CopyOnlyVector<int>>;
    M m1    = M({1, 1, 3}, {1, 2, 3});
    M m2(std::move(m1), std::allocator<int>{});
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.keys().size() == 0);
    LIBCPP_ASSERT(m1.values().size() == 0);
  }

  return 0;
}
