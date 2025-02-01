//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set(initializer_list<value_type> il, const key_compare& comp = key_compare());
// template<class Alloc>
//    flat_set(initializer_list<value_type> il, const Alloc& a);
// template<class Alloc>
//    flat_set(initializer_list<value_type> il, const key_compare& comp, const Alloc& a);

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

#include "../../../test_compare.h"

struct DefaultCtableComp {
  explicit DefaultCtableComp() { default_constructed_ = true; }
  bool operator()(int, int) const { return false; }
  bool default_constructed_ = false;
};

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
    using IL = std::initializer_list<int>;
    static_assert(std::is_constructible_v<M1, IL, const A1&>);
    static_assert(std::is_constructible_v<M2, IL, const A2&>);
    static_assert(!std::is_constructible_v<M1, IL, const A2&>);
    static_assert(!std::is_constructible_v<M2, IL, const A1&>);

    static_assert(std::is_constructible_v<M1, IL, const C&, const A1&>);
    static_assert(std::is_constructible_v<M2, IL, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M1, IL, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, IL, const C&, const A1&>);
  }

  {
    // initializer_list<value_type> needs to match exactly
    using M = std::flat_set<int>;
    using C = typename M::key_compare;
    static_assert(std::is_constructible_v<M, std::initializer_list<int>>);
    static_assert(std::is_constructible_v<M, std::initializer_list<int>, C>);
    static_assert(std::is_constructible_v<M, std::initializer_list<int>, C, std::allocator<int>>);
    static_assert(std::is_constructible_v<M, std::initializer_list<int>, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>, C>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>, C, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>, C>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>, C, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<const int>, std::allocator<int>>);
  }

  int expected[] = {1, 2, 3, 5};
  {
    // flat_set(initializer_list<value_type>);
    using M                       = std::flat_set<int>;
    std::initializer_list<int> il = {5, 2, 2, 3, 1, 3};
    M m(il);
    assert(std::equal(m.begin(), m.end(), expected, expected + 4));
  }
  {
    // flat_set(initializer_list<value_type>);
    // explicit(false)
    using M = std::flat_set<int>;
    M m     = {5, 2, 2, 3, 1, 3};
    assert(std::equal(m.begin(), m.end(), expected, expected + 4));
  }
  {
    // flat_set(initializer_list<value_type>);
    using M = std::flat_set<int, std::greater<int>, std::deque<int, min_allocator<int>>>;
    M m     = {5, 2, 2, 3, 1, 3};
    assert(std::equal(m.rbegin(), m.rend(), expected, expected + 4));
  }
  {
    using A = explicit_allocator<int>;
    {
      // flat_set(initializer_list<value_type>);
      // different comparator
      using M = std::flat_set<int, DefaultCtableComp, std::vector<int, A>>;
      M m     = {1, 2, 3};
      assert(m.size() == 1);
      LIBCPP_ASSERT(*m.begin() == 1);
      assert(m.key_comp().default_constructed_);
    }
    {
      // flat_set(initializer_list<value_type>, const Allocator&);
      using M = std::flat_set<int, std::greater<int>, std::deque<int, A>>;
      A a;
      M m({5, 2, 2, 3, 1, 3}, a);
      assert(std::equal(m.rbegin(), m.rend(), expected, expected + 4));
    }
  }
  {
    // flat_set(initializer_list<value_type>, const key_compare&);
    using C = test_less<int>;
    using M = std::flat_set<int, C>;
    auto m  = M({5, 2, 2, 3, 1, 3}, C(10));
    assert(std::equal(m.begin(), m.end(), expected, expected + 4));
    assert(m.key_comp() == C(10));

    // explicit(false)
    M m2 = {{5, 2, 2, 1, 3, 3}, C(10)};
    assert(m2 == m);
    assert(m2.key_comp() == C(10));
  }
  {
    // flat_set(initializer_list<value_type>, const key_compare&);
    // Sorting uses the comparator that was passed in
    using M = std::flat_set<int, std::function<bool(int, int)>, std::deque<int, min_allocator<int>>>;
    auto m  = M({5, 2, 2, 1, 3, 1}, std::greater<int>());
    assert(std::equal(m.rbegin(), m.rend(), expected, expected + 4));
    assert(m.key_comp()(2, 1) == true);
  }
  {
    // flat_set(initializer_list<value_type> il, const key_compare& comp, const Alloc& a);
    using A = explicit_allocator<int>;
    using M = std::flat_set<int, std::greater<int>, std::deque<int, A>>;
    A a;
    M m({5, 2, 2, 3, 1, 3}, {}, a);
    assert(std::equal(m.rbegin(), m.rend(), expected, expected + 4));
  }

  return 0;
}
