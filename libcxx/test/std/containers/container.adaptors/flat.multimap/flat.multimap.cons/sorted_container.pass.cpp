//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(sorted_equivalent_t, key_container_type key_cont, mapped_container_type mapped_cont,
//          const key_compare& comp = key_compare());
//
// template<class Alloc>
//   flat_multimap(sorted_equivalent_t, const key_container_type& key_cont,
//            const mapped_container_type& mapped_cont, const Alloc& a);
// template<class Alloc>
//   flat_multimap(sorted_equivalent_t, const key_container_type& key_cont,
//            const mapped_container_type& mapped_cont,
//            const key_compare& comp, const Alloc& a);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../helpers.h"
#include "../../../test_compare.h"

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    // flat_multimap(sorted_equivalent_t, key_container_type , mapped_container_type)
    using M                 = std::flat_multimap<int, char, std::less<int>, KeyContainer<int>, ValueContainer<char>>;
    KeyContainer<int> ks    = {1, 4, 4, 10};
    ValueContainer<char> vs = {4, 3, 2, 1};
    auto ks2                = ks;
    auto vs2                = vs;

    auto m = M(std::sorted_equivalent, ks, vs);
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{1, 4}, {4, 3}, {4, 2}, {10, 1}}));
    m = M(std::sorted_equivalent, std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{1, 4}, {4, 3}, {4, 2}, {10, 1}}));

    // explicit(false)
    M m2 = {std::sorted_equivalent, std::move(ks2), std::move(vs2)};
    assert(m == m2);
  }
  {
    // flat_multimap(sorted_equivalent_t, key_container_type , mapped_container_type)
    // non-default container, comparator and allocator type
    using Ks = KeyContainer<int, min_allocator<int>>;
    using Vs = ValueContainer<char, min_allocator<char>>;
    using M  = std::flat_multimap<int, char, std::greater<int>, Ks, Vs>;
    Ks ks    = {10, 4, 4, 1};
    Vs vs    = {1, 2, 3, 4};
    auto m   = M(std::sorted_equivalent, ks, vs);
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{10, 1}, {4, 2}, {4, 3}, {1, 4}}));
    m = M(std::sorted_equivalent, std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{10, 1}, {4, 2}, {4, 3}, {1, 4}}));
  }
  {
    // flat_multimap(sorted_equivalent_t, key_container_type , mapped_container_type)
    // allocator copied into the containers
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, KeyContainer<int, A>, ValueContainer<int, A>>;
    auto ks = KeyContainer<int, A>({1, 4, 4, 10}, A(4));
    auto vs = ValueContainer<int, A>({4, 3, 2, 1}, A(5));
    auto m  = M(std::sorted_equivalent, std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{1, 4}, {4, 3}, {4, 2}, {10, 1}}));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(5));
  }
  {
    // flat_multimap(sorted_equivalent_t, key_container_type , mapped_container_type, key_compare)
    using C                 = test_less<int>;
    using M                 = std::flat_multimap<int, char, C, KeyContainer<int>, ValueContainer<char>>;
    KeyContainer<int> ks    = {1, 4, 4, 10};
    ValueContainer<char> vs = {4, 3, 2, 1};

    auto m = M(std::sorted_equivalent, ks, vs, C(4));
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{1, 4}, {4, 3}, {4, 2}, {10, 1}}));
    assert(m.key_comp() == C(4));

    // explicit(false)
    M m2 = {std::sorted_equivalent, ks, vs, C(4)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
  }
  {
    // flat_multimap(sorted_equivalent_t, key_container_type , mapped_container_type, key_compare, const Allocator&)
    using C                   = test_less<int>;
    using A                   = test_allocator<int>;
    using M                   = std::flat_multimap<int, int, C, KeyContainer<int, A>, ValueContainer<int, A>>;
    KeyContainer<int, A> ks   = {1, 4, 4, 10};
    ValueContainer<int, A> vs = {4, 3, 2, 1};
    auto m                    = M(std::sorted_equivalent, ks, vs, C(4), A(5));
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{1, 4}, {4, 3}, {4, 2}, {10, 1}}));
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A(5));
    assert(m.values().get_allocator() == A(5));

    // explicit(false)
    M m2 = {std::sorted_equivalent, ks, vs, C(4), A(5)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
    assert(m2.keys().get_allocator() == A(5));
    assert(m2.values().get_allocator() == A(5));
  }
  {
    // flat_multimap(sorted_equivalent_t, key_container_type , mapped_container_type, const Allocator&)
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, KeyContainer<int, A>, ValueContainer<int, A>>;
    auto ks = KeyContainer<int, A>({1, 4, 4, 10}, A(4));
    auto vs = ValueContainer<int, A>({4, 3, 2, 1}, A(5));
    auto m  = M(std::sorted_equivalent, ks, vs, A(6)); // replaces the allocators
    assert(!ks.empty());                               // it was an lvalue above
    assert(!vs.empty());                               // it was an lvalue above
    assert(std::ranges::equal(m, std::vector<std::pair<int, char>>{{1, 4}, {4, 3}, {4, 2}, {10, 1}}));
    assert(m.keys().get_allocator() == A(6));
    assert(m.values().get_allocator() == A(6));

    // explicit(false)
    M m2 = {std::sorted_equivalent, ks, vs, A(6)};
    assert(m2 == m);
    assert(m2.keys().get_allocator() == A(6));
    assert(m2.values().get_allocator() == A(6));
  }
}

constexpr bool test() {
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
    static_assert(std::is_constructible_v<M1, std::sorted_equivalent_t, const V1&, const V1&, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_equivalent_t, const V1&, const V1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_equivalent_t, const V1&, const V2&, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::sorted_equivalent_t, const V2&, const V1&, const A2&>);

    static_assert(std::is_constructible_v<M1, std::sorted_equivalent_t, const V1&, const V1&, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_equivalent_t, const V1&, const V1&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_equivalent_t, const V1&, const V2&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::sorted_equivalent_t, const V2&, const V1&, const C&, const A2&>);
  }

  test<std::vector, std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque, std::vector>();
    test<std::deque, std::deque>();
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
