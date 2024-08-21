//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(initializer_list<value_type> il, const key_compare& comp = key_compare());
// template<class Alloc> flat_map(initializer_list<value_type> il, const key_compare& comp, const Alloc& a);

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "min_allocator.h"
#include "../../../test_compare.h"

int main(int, char**) {
  std::pair<int, short> expected[] = {{1, 1}, {2, 2}, {3, 3}, {5, 2}};
  {
    using C = test_less<int>;
    using M = std::flat_map<int, short, C>;
    auto m  = M({{5, 2}, {2, 2}, {2, 2}, {3, 3}, {1, 1}, {3, 3}}, C(10));
    assert(std::equal(m.begin(), m.end(), expected, expected + 4));
    assert(m.key_comp() == C(10));
  }
  {
    // Sorting uses the comparator that was passed in
    using M = std::flat_map<int, short, std::function<bool(int, int)>, std::deque<int, min_allocator<int>>>;
    auto m  = M({{5, 2}, {2, 2}, {2, 2}, {3, 3}, {1, 1}, {3, 3}}, std::greater<int>());
    assert(std::equal(m.rbegin(), m.rend(), expected, expected + 4));
    assert(m.key_comp()(2, 1) == true);
  }
  {
    using C                                 = test_less<int>;
    using M                                 = std::flat_map<int, short, C>;
    std::initializer_list<M::value_type> il = {{5, 2}, {2, 2}, {2, 2}, {3, 3}, {1, 1}, {3, 3}};
    auto m                                  = M(il, C(10));
    assert(std::equal(m.begin(), m.end(), expected, expected + 4));
    assert(m.key_comp() == C(10));
    static_assert(std::is_constructible_v<M, std::initializer_list<std::pair<int, short>>, C>);
    static_assert(std::is_constructible_v<M, std::initializer_list<std::pair<int, short>>, C, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<std::pair<const int, short>>, C>);
    static_assert(
        !std::is_constructible_v<M, std::initializer_list<std::pair<const int, short>>, C, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<std::pair<const int, const short>>, C>);
    static_assert(
        !std::is_constructible_v<M, std::initializer_list<std::pair<const int, const short>>, C, std::allocator<int>>);
  }
  {
    using A = explicit_allocator<int>;
    using M = std::flat_map<int, int, std::greater<int>, std::deque<int, A>, std::vector<int, A>>;
    A a;
    M m({{5, 2}, {2, 2}, {2, 2}, {3, 3}, {1, 1}, {3, 3}}, {}, a);
    assert(std::equal(m.rbegin(), m.rend(), expected, expected + 4));
  }
  {
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::deque<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::initializer_list<M::value_type> il = {{3, 3}, {1, 1}, {4, 4}, {1, 1}, {5, 5}};
    vm.emplace_back(il, C(5));
    assert((vm[0] == M{{1, 1}, {3, 3}, {4, 4}, {5, 5}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
    assert(vm[0].key_comp() == C(5));
  }
  return 0;
}
