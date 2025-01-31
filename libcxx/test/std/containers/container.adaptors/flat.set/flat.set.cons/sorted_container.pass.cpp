//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set(sorted_unique_t, container_type key_cont, const key_compare& comp = key_compare());
//
// template<class Alloc>
//   flat_set(sorted_unique_t, const container_type& key_cont, const Alloc& a);
// template<class Alloc>
//   flat_set(sorted_unique_t, const container_type& key_cont,
//            const key_compare& comp, const Alloc& a);

#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../../../test_compare.h"

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
    static_assert(std::is_constructible_v<M1, std::sorted_unique_t, const V1&, const A1&>);
    static_assert(std::is_constructible_v<M2, std::sorted_unique_t, const V2&, const A2&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_unique_t, const V1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_unique_t, const V2&, const A1&>);

    static_assert(std::is_constructible_v<M1, std::sorted_unique_t, const V1&, const C&, const A1&>);
    static_assert(std::is_constructible_v<M2, std::sorted_unique_t, const V2&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_unique_t, const V1&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_unique_t, const V2&, const C&, const A1&>);
  }
  {
    // flat_set(sorted_unique_t, container_type)
    using M             = std::flat_set<int>;
    std::vector<int> ks = {1, 2, 4, 10};
    auto ks2            = ks;

    auto m = M(std::sorted_unique, ks);
    assert((m == M{1, 2, 4, 10}));
    m = M(std::sorted_unique, std::move(ks));
    assert(ks.empty()); // it was moved-from
    assert((m == M{1, 2, 4, 10}));

    // explicit(false)
    M m2 = {std::sorted_unique, std::move(ks2)};
    assert(m == m2);
  }
  {
    // flat_set(sorted_unique_t, container_type)
    // non-default container, comparator and allocator type
    using Ks = std::deque<int, min_allocator<int>>;
    using M  = std::flat_set<int, std::greater<int>, Ks>;
    Ks ks    = {10, 4, 2, 1};
    auto m   = M(std::sorted_unique, ks);
    assert((m == M{1, 2, 4, 10}));
    m = M(std::sorted_unique, std::move(ks));
    assert(ks.empty()); // it was moved-from
    assert((m == M{1, 2, 4, 10}));
  }
  {
    // flat_set(sorted_unique_t, container_type)
    // allocator copied into the containers
    using A = test_allocator<int>;
    using M = std::flat_set<int, std::less<int>, std::deque<int, A>>;
    auto ks = std::deque<int, A>({1, 2, 4, 10}, A(4));
    auto m  = M(std::sorted_unique, std::move(ks));
    assert(ks.empty()); // it was moved-from
    assert((m == M{1, 2, 4, 10}));
    assert(std::move(m).extract().get_allocator() == A(4));
  }
  {
    // flat_set(sorted_unique_t, container_type ,  key_compare)
    using C             = test_less<int>;
    using M             = std::flat_set<int, C>;
    std::vector<int> ks = {1, 2, 4, 10};

    auto m = M(std::sorted_unique, ks, C(4));
    assert((m == M{1, 2, 4, 10}));
    assert(m.key_comp() == C(4));

    // explicit(false)
    M m2 = {std::sorted_unique, ks, C(4)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
  }
  {
    // flat_set(sorted_unique_t, container_type , key_compare, const Allocator&)
    using C                = test_less<int>;
    using A                = test_allocator<int>;
    using M                = std::flat_set<int, C, std::vector<int, A>>;
    std::vector<int, A> ks = {1, 2, 4, 10};
    auto m                 = M(std::sorted_unique, ks, C(4), A(5));
    assert((m == M{1, 2, 4, 10}));
    assert(m.key_comp() == C(4));
    assert(M(m).extract().get_allocator() == A(5));

    // explicit(false)
    M m2 = {ks, C(4), A(5)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
    assert(std::move(m2).extract().get_allocator() == A(5));
  }
  {
    // flat_set(sorted_unique_t, container_type , const Allocator&)
    using A = test_allocator<int>;
    using M = std::flat_set<int, std::less<int>, std::deque<int, A>>;
    auto ks = std::deque<int, A>({1, 2, 4, 10}, A(4));
    auto m  = M(std::sorted_unique, ks, A(6)); // replaces the allocators
    assert(!ks.empty());                       // it was an lvalue above
    assert((m == M{1, 2, 4, 10}));
    assert(M(m).extract().get_allocator() == A(6));

    // explicit(false)
    M m2 = {std::sorted_unique, ks, A(6)};
    assert(m2 == m);
    assert(std::move(m2).extract().get_allocator() == A(6));
  }

  return 0;
}
