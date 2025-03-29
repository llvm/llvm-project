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
#include "../../../test_compare.h"

struct P {
  int first;
  int second;
  template <class T, class U>
  bool operator==(const std::pair<T, U>& rhs) const {
    return MoveOnly(first) == rhs.first && MoveOnly(second) == rhs.second;
  }
};

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
    static_assert(std::is_constructible_v<M1, const V1&, const V1&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const V1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V1&, const V2&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const V2&, const V1&, const A2&>);

    static_assert(std::is_constructible_v<M1, const V1&, const V1&, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const V1&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V1&, const V2&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const V2&, const V1&, const C&, const A2&>);
  }
  {
    // flat_multimap(key_container_type , mapped_container_type)
    using M                         = std::flat_multimap<int, char>;
    std::vector<int> ks             = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<char> vs            = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                          = M(ks, vs);
    std::pair<int, char> expected[] = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {2, 7}, {3, 6}, {3, 8}, {3, 9}};
    assert(std::ranges::equal(m, expected));

    // explicit(false)
    M m2 = {ks, vs};
    assert(m2 == m);

    m = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m, expected));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type)
    // move-only
    P expected[] = {{3, 3}, {3, 2}, {2, 1}, {1, 4}};
    using Ks     = std::deque<int, min_allocator<int>>;
    using Vs     = std::vector<MoveOnly, min_allocator<MoveOnly>>;
    using M      = std::flat_multimap<int, MoveOnly, std::greater<int>, Ks, Vs>;
    Ks ks        = {1, 3, 3, 2};
    Vs vs;
    vs.push_back(4);
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
    using M = std::flat_multimap<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    auto m  = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    std::pair<int, int> expected[] = {{1, 1}, {1, 1}, {1, 1}, {2, 2}, {2, 2}, {2, 2}, {3, 3}, {3, 3}, {3, 3}};
    assert(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A(5));
    assert(m.values().get_allocator() == A(6));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, key_compare)
    using C                         = test_less<int>;
    using M                         = std::flat_multimap<int, char, C>;
    std::vector<int> ks             = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<char> vs            = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                          = M(ks, vs, C(4));
    std::pair<int, char> expected[] = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {2, 7}, {3, 6}, {3, 8}, {3, 9}};
    assert(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(4));

    // explicit(false)
    M m2 = {ks, vs, C(4)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, const Allocator&)
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    auto m  = M(ks, vs, A(4)); // replaces the allocators
    assert(!ks.empty());       // it was an lvalue above
    assert(!vs.empty());       // it was an lvalue above
    std::pair<int, int> expected[] = {{1, 1}, {1, 1}, {1, 1}, {2, 2}, {2, 2}, {2, 2}, {3, 3}, {3, 3}, {3, 3}};
    assert(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(4));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, const Allocator&)
    // explicit(false)
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    M m     = {ks, vs, A(4)}; // implicit ctor
    assert(!ks.empty());      // it was an lvalue above
    assert(!vs.empty());      // it was an lvalue above
    std::pair<int, int> expected[] = {{1, 1}, {1, 1}, {1, 1}, {2, 2}, {2, 2}, {2, 2}, {3, 3}, {3, 3}, {3, 3}};
    assert(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(4));
  }
  {
    // flat_multimap(key_container_type , mapped_container_type, key_compare, const Allocator&)
    using C                         = test_less<int>;
    using A                         = test_allocator<int>;
    using M                         = std::flat_multimap<int, int, C, std::vector<int, A>, std::vector<int, A>>;
    std::vector<int, A> ks          = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<int, A> vs          = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                          = M(ks, vs, C(4), A(5));
    std::pair<int, char> expected[] = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {2, 7}, {3, 6}, {3, 8}, {3, 9}};
    assert(std::ranges::equal(m, expected));
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

  return 0;
}
