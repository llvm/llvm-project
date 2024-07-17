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
#include <memory_resource>
#include <string>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
  {
    using C = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<char>;
    using M = std::flat_map<int, char, C, std::vector<int, A1>, std::vector<char, A2>>;
    M mo = M({{1,1},{2,3},{3,2}}, C(5), A1(7));
    M m = M({}, C(3), A1(7));
    m = std::move(mo);
    assert((m == M{{1,1},{2,3},{3,2}}));
    assert(m.key_comp() == C(5));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(vs.get_allocator() == A2(7));
    assert(mo.empty());
  }
  {
    using C = test_less<int>;
    using A1 = other_allocator<int>;
    using A2 = other_allocator<char>;
    using M = std::flat_map<int, char, C, std::deque<int, A1>, std::deque<char, A2>>;
    M mo = M({{4,5},{5,4}}, C(5), A1(7));
    M m = M({{1,1},{2,2},{3,3},{4,4}}, C(3), A1(7));
    m = std::move(mo);
    assert((m == M{{4,5},{5,4}}));
    assert(m.key_comp() == C(5));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(vs.get_allocator() == A2(7));
    assert(mo.empty());
  }
  {
    using A = min_allocator<int>;
    using M = std::flat_map<int, int, std::greater<int>, std::vector<int, A>, std::vector<int, A>>;
    M mo = M({{5,1},{4,2},{3,3}}, A());
    M m = M({{4,4},{3,3},{2,2},{1,1}}, A());
    m = std::move(mo);
    assert((m == M{{5,1},{4,2},{3,3}}));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A());
    assert(vs.get_allocator() == A());
    assert(mo.empty());
  }
  {
    // A moved-from flat_map maintains its class invariant in the presence of moved-from elements.
    using M = std::flat_map<std::pmr::string, int, std::less<>, std::pmr::vector<std::pmr::string>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({{"short", 1}, {"very long string that definitely won't fit in the SSO buffer and therefore becomes empty on move", 2}}, &mr1);
    M m = M({{"don't care", 3}}, &mr2);
    m = std::move(mo);
    assert(m.size() == 2);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.begin()->first.get_allocator().resource() == &mr2);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    mo.insert({"foo",1});
    assert(mo.begin()->first.get_allocator().resource() == &mr1);
  }
  {
    // A moved-from flat_map maintains its class invariant in the presence of moved-from comparators.
    using C = std::function<bool(int,int)>;
    using M = std::flat_map<int, int, C>;
    M mo = M({{1,3},{2,2},{3,1}}, std::less<int>());
    M m = M({{1,1},{2,2}}, std::greater<int>());
    m = std::move(mo);
    assert(m.size() == 3);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.key_comp()(1,2) == true);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    LIBCPP_ASSERT(m.key_comp()(1,2) == true);
    LIBCPP_ASSERT(mo.empty());
    mo.insert({{1,3},{2,2},{3,1}}); // insert has no preconditions
    LIBCPP_ASSERT(m == mo);
  }
  return 0;
}
