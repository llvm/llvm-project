//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// explicit flat_multiset(container_type key_cont, const key_compare& comp = key_compare());
// template<class Allocator>
//   flat_multiset(const container_type& key_cont, const Allocator& a);
// template<class Alloc>
//   flat_multiset(const container_type& key_cont, const key_compare& comp, const Alloc& a);

#include <algorithm>
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

template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };

void test() {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<container_type, Alloc> is true

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using V1 = std::vector<int, A1>;
    using V2 = std::vector<int, A2>;
    using M1 = std::flat_multiset<int, C, V1>;
    using M2 = std::flat_multiset<int, C, V2>;
    static_assert(std::is_constructible_v<M1, const V1&, const A1&>);
    static_assert(std::is_constructible_v<M2, const V2&, const A2&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V2&, const A1&>);

    static_assert(std::is_constructible_v<M1, const V1&, const C&, const A1&>);
    static_assert(std::is_constructible_v<M2, const V2&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V2&, const C&, const A1&>);
  }
  {
    // flat_multiset(container_type)
    using M             = std::flat_multiset<int>;
    std::vector<int> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    auto m              = M(ks);
    int expected[]      = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    assert(std::ranges::equal(m, expected));

    // explicit(false)
    static_assert(std::is_constructible_v<M, const std::vector<int>&>);
    static_assert(!ImplicitlyConstructible<M, const std::vector<int>&>);

    m = M(std::move(ks));
    assert(ks.empty()); // it was moved-from
    assert(std::ranges::equal(m, expected));
  }
  {
    // flat_multiset(container_type)
    // move-only
    int expected[] = {3, 3, 2, 1};
    using Ks       = std::deque<MoveOnly, min_allocator<MoveOnly>>;
    using M        = std::flat_multiset<MoveOnly, std::greater<MoveOnly>, Ks>;
    Ks ks;
    ks.push_back(1);
    ks.push_back(3);
    ks.push_back(3);
    ks.push_back(2);
    auto m = M(std::move(ks));
    assert(ks.empty()); // it was moved-from
    assert(std::ranges::equal(m, expected, std::equal_to<>()));
  }
  {
    // flat_multiset(container_type)
    // container's allocators are used
    using A = test_allocator<int>;
    using M = std::flat_multiset<int, std::less<int>, std::deque<int, A>>;
    auto ks = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto m  = M(std::move(ks));
    assert(ks.empty()); // it was moved-from
    assert((m == M{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    auto keys = std::move(m).extract();
    assert(keys.get_allocator() == A(5));
  }
  {
    // flat_multiset(container_type, key_compare)
    using C             = test_less<int>;
    using M             = std::flat_multiset<int, C>;
    std::vector<int> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    auto m              = M(ks, C(4));
    assert(std::ranges::equal(m, std::vector<int>{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(m.key_comp() == C(4));

    // explicit
    static_assert(std::is_constructible_v<M, const std::vector<int>&, const C&>);
    static_assert(!ImplicitlyConstructible<M, const std::vector<int>&, const C&>);
  }
  {
    // flat_multiset(container_type , const Allocator&)
    using A = test_allocator<int>;
    using M = std::flat_multiset<int, std::less<int>, std::deque<int, A>>;
    auto ks = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto m  = M(ks, A(4)); // replaces the allocators
    assert(!ks.empty());   // it was an lvalue above
    assert((m == M{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    auto keys = M(m).extract();
    assert(keys.get_allocator() == A(4));

    // explicit(false)
    static_assert(ImplicitlyConstructible<M, const std::deque<int, A>&, const A&>);
    M m2 = {ks, A(4)};   // implicit ctor
    assert(!ks.empty()); // it was an lvalue above
    assert(m2 == m);
    auto keys2 = std::move(m).extract();
    assert(keys2.get_allocator() == A(4));
  }
  {
    // flat_multiset(container_type , const Allocator&)
    using C                = test_less<int>;
    using A                = test_allocator<int>;
    using M                = std::flat_multiset<int, C, std::vector<int, A>>;
    std::vector<int, A> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    auto m                 = M(ks, C(4), A(5));
    assert(std::ranges::equal(m, std::vector<int, A>{1, 1, 1, 2, 2, 2, 3, 3, 3}));
    assert(m.key_comp() == C(4));
    auto m_copy = m;
    auto keys   = std::move(m_copy).extract();
    assert(keys.get_allocator() == A(5));

    // explicit(false)
    static_assert(ImplicitlyConstructible<M, const std::vector<int, A>&, const A&>);
    M m2 = {ks, C(4), A(5)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
    keys = std::move(m2).extract();
    assert(keys.get_allocator() == A(5));
  }
}

int main(int, char**) {
  test();

  return 0;
}
