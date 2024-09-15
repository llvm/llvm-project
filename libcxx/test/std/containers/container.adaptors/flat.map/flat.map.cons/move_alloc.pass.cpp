//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(flat_map&&, const allocator_type&);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <ranges>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 1}};
    using C                        = test_less<int>;
    using A                        = test_allocator<int>;
    using M                        = std::flat_map<int, int, C, std::vector<int, A>, std::deque<int, A>>;
    auto mo                        = M(expected, expected + 3, C(5), A(7));
    auto m                         = M(std::move(mo), A(3));

    assert(m.key_comp() == C(5));
    assert(m.size() == 3);
    auto [keys, values] = std::move(m).extract();
    assert(keys.get_allocator() == A(3));
    assert(values.get_allocator() == A(3));
    assert(std::ranges::equal(keys, expected | std::views::elements<0>));
    assert(std::ranges::equal(values, expected | std::views::elements<1>));

    // The original flat_map is moved-from.
    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(mo.keys().get_allocator() == A(7));
    assert(mo.values().get_allocator() == A(7));
  }
  {
    std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 1}};
    using C                        = test_less<int>;
    using M                        = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::deque<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({{1, 1}, {3, 1}, {1, 1}, {2, 2}}, C(5), &mr1);
    M m  = {std::move(mo), &mr2}; // also test the implicitness of this constructor

    assert(m.key_comp() == C(5));
    assert(m.size() == 3);
    assert(m.keys().get_allocator().resource() == &mr2);
    assert(m.values().get_allocator().resource() == &mr2);
    assert(std::equal(m.begin(), m.end(), expected, expected + 3));

    // The original flat_map is moved-from.
    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    assert(mo.key_comp() == C(5));
    assert(mo.keys().get_allocator().resource() == &mr1);
    assert(mo.values().get_allocator().resource() == &mr1);
  }
  {
    using M = std::flat_map<int, int, std::less<>, std::pmr::deque<int>, std::pmr::vector<int>>;
    std::pmr::vector<M> vs;
    M m = {{1, 1}, {3, 1}, {1, 1}, {2, 2}};
    vs.push_back(std::move(m));
    assert((vs[0].keys() == std::pmr::deque<int>{1, 2, 3}));
    assert((vs[0].values() == std::pmr::vector<int>{1, 2, 1}));
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_map<int, int, std::less<>, std::vector<int>, CopyOnlyVector<int>>;
    M m1    = M({1,2,3},{1,2,3});
    M m2 (std::move(m1), std::allocator<int>{});
    assert(m2.size()==3);
    assert(m1.keys().size() == m1.values().size());
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.keys().size() == 0);
    LIBCPP_ASSERT(m1.values().size() == 0);
  }
  return 0;
}
