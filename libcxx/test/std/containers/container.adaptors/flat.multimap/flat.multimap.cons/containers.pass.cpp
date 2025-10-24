//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(key_container_type key_cont, mapped_container_type mapped_cont,
//           const key_compare& comp = key_compare());
// template<class Allocator>
//   flat_multimap(const key_container_type& key_cont, const mapped_container_type& mapped_cont,
//            const Allocator& a);
// template<class Alloc>
//   flat_multimap(const key_container_type& key_cont, const mapped_container_type& mapped_cont,
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

struct P {
  int first;
  int second;
  template <class T, class U>
  constexpr bool operator==(const std::pair<T, U>& rhs) const {
    return MoveOnly(first) == rhs.first && MoveOnly(second) == rhs.second;
  }
};

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    // flat_multimap(key_container_type , mapped_container_type)
    using M                  = std::flat_multimap<int, short, std::less<int>, KeyContainer<int>, ValueContainer<short>>;
    KeyContainer<int> ks     = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    ValueContainer<short> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                   = M(ks, vs);
    assert((m.keys() == KeyContainer<int>{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    check_possible_values(
        m.values(),
        std::vector<std::vector<short>>{
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {4, 5, 7},
            {4, 5, 7},
            {4, 5, 7},
            {6, 8, 9},
            {6, 8, 9},
            {6, 8, 9},
        });

    // explicit(false)
    M m2 = {ks, vs};
    assert(m2 == m);

    m = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert((m.keys() == KeyContainer<int>{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    check_possible_values(
        m.values(),
        std::vector<std::vector<short>>{
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {4, 5, 7},
            {4, 5, 7},
            {4, 5, 7},
            {6, 8, 9},
            {6, 8, 9},
            {6, 8, 9},
        });
  }
  {
    // flat_multimap(key_container_type , mapped_container_type)
    // move-only
    P expected[] = {{3, 2}, {2, 1}, {1, 3}};
    using Ks     = KeyContainer<int, min_allocator<int>>;
    using Vs     = ValueContainer<MoveOnly, min_allocator<MoveOnly>>;
    using M      = std::flat_multimap<int, MoveOnly, std::greater<int>, Ks, Vs>;
    Ks ks        = {1, 3, 2};
    Vs vs;
    vs.push_back(3);
    vs.push_back(2);
    vs.push_back(1);
    auto m = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m, expected, std::equal_to<>()));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type)
    // container's allocators are used
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, KeyContainer<int, A>, ValueContainer<int, A>>;
    auto ks = KeyContainer<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = ValueContainer<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    auto m  = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m.keys(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(std::ranges::equal(m.values(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(m.keys().get_allocator() == A(5));
    assert(m.values().get_allocator() == A(6));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, key_compare)
    using C                 = test_less<int>;
    using M                 = std::flat_multimap<int, char, C, KeyContainer<int>, ValueContainer<char>>;
    KeyContainer<int> ks    = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    ValueContainer<char> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                  = M(ks, vs, C(4));
    assert((m.keys() == KeyContainer<int>{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    check_possible_values(
        m.values(),
        std::vector<std::vector<char>>{
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {4, 5, 7},
            {4, 5, 7},
            {4, 5, 7},
            {6, 8, 9},
            {6, 8, 9},
            {6, 8, 9},
        });
    assert(m.key_comp() == C(4));

    // explicit(false)
    M m2 = {ks, vs, C(4)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, const Allocator&)
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, KeyContainer<int, A>, ValueContainer<int, A>>;
    auto ks = KeyContainer<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = ValueContainer<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    auto m  = M(ks, vs, A(4)); // replaces the allocators
    assert(!ks.empty());       // it was an lvalue above
    assert(!vs.empty());       // it was an lvalue above
    assert(std::ranges::equal(m.keys(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(std::ranges::equal(m.values(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(4));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, const Allocator&)
    // explicit(false)
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, KeyContainer<int, A>, ValueContainer<int, A>>;
    auto ks = KeyContainer<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = ValueContainer<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    M m     = {ks, vs, A(4)}; // implicit ctor
    assert(!ks.empty());      // it was an lvalue above
    assert(!vs.empty());      // it was an lvalue above
    assert(std::ranges::equal(m.keys(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(std::ranges::equal(m.values(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(4));
  }

  {
    // flat_multimap(key_container_type , mapped_container_type, key_compare, const Allocator&)
    using C                = test_less<int>;
    using A                = test_allocator<int>;
    using M                = std::flat_multimap<int, int, C, std::vector<int, A>, std::vector<int, A>>;
    std::vector<int, A> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<int, A> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                 = M(ks, vs, C(4), A(5));
    assert(std::ranges::equal(m.keys(), std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    check_possible_values(
        m.values(),
        std::vector<std::vector<int>>{
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {4, 5, 7},
            {4, 5, 7},
            {4, 5, 7},
            {6, 8, 9},
            {6, 8, 9},
            {6, 8, 9},
        });
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A(5));
    assert(m.values().get_allocator() == A(5));

    // explicit(false)
    M m2 = {ks, vs, C(4), A(5)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
    assert(m2.keys().get_allocator() == A(5));
    assert(m2.values().get_allocator() == A(5));
  }
}

bool constexpr test() {
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
    static_assert(std::is_constructible_v<M1, const V1&, const V1&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const V1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V1&, const V2&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const V2&, const V1&, const A2&>);

    static_assert(std::is_constructible_v<M1, const V1&, const V1&, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const V1&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V1&, const V2&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const V2&, const V1&, const C&, const A2&>);
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
