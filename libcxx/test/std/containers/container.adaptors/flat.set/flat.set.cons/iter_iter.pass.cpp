//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template <class InputIterator>
//   flat_set(InputIterator first, InputIterator last, const key_compare& comp = key_compare());
// template<class InputIterator, class Allocator>
//   flat_set(InputIterator first, InputIterator last, const Allocator& a);
// template<class InputIterator, class Allocator>
//   flat_set(InputIterator first, InputIterator last, const key_compare& comp, const Allocator& a);

#include <algorithm>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../../../test_compare.h"

int main(int, char**) {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<container_type, Alloc> is true.

    using C     = test_less<int>;
    using A1    = test_allocator<int>;
    using A2    = other_allocator<int>;
    using V1    = std::vector<int, A1>;
    using V2    = std::vector<int, A2>;
    using M1    = std::flat_set<int, C, V1>;
    using M2    = std::flat_set<int, C, V2>;
    using Iter1 = typename M1::iterator;
    using Iter2 = typename M2::iterator;
    static_assert(std::is_constructible_v<M1, Iter1, Iter1, const A1&>);
    static_assert(std::is_constructible_v<M2, Iter2, Iter2, const A2&>);
    static_assert(!std::is_constructible_v<M1, Iter1, Iter1, const A2&>);
    static_assert(!std::is_constructible_v<M2, Iter2, Iter2, const A1&>);

    static_assert(std::is_constructible_v<M1, Iter1, Iter1, const C&, const A1&>);
    static_assert(std::is_constructible_v<M2, Iter2, Iter2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M1, Iter1, Iter1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, Iter2, Iter2, const C&, const A1&>);
  }

  int ar[]       = {1, 1, 1, 2, 2, 3, 2, 3, 3};
  int expected[] = {1, 2, 3};
  {
    // flat_set(InputIterator , InputIterator)
    // cpp17_input_iterator
    using M = std::flat_set<int>;
    auto m  = M(cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 9));
    assert(std::ranges::equal(m, expected));

    // explicit(false)
    M m2 = {cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 9)};
    assert(m2 == m);
  }
  {
    // flat_set(InputIterator , InputIterator)
    // greater
    using M = std::flat_set<int, std::greater<int>, std::deque<int, min_allocator<int>>>;
    auto m  = M(cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 9));
    assert(std::ranges::equal(m, std::deque<int, min_allocator<int>>{3, 2, 1}));
  }
  {
    // flat_set(InputIterator , InputIterator)
    // Test when the operands are of array type (also contiguous iterator type)
    using M = std::flat_set<int, std::greater<int>, std::vector<int, min_allocator<int>>>;
    auto m  = M(ar, ar);
    assert(m.empty());
  }
  {
    // flat_set(InputIterator , InputIterator, const key_compare&)
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::vector<int>>;
    auto m  = M(ar, ar + 9, C(3));
    assert(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));

    // explicit(false)
    M m2 = {ar, ar + 9, C(3)};
    assert(m2 == m);
    assert(m2.key_comp() == C(3));
  }
  {
    // flat_set(InputIterator , InputIterator, const Allocator&)
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, std::less<int>, std::vector<int, A1>>;
    auto m   = M(ar, ar + 9, A1(5));
    assert(std::ranges::equal(m, expected));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_set(InputIterator , InputIterator, const Allocator&)
    // explicit(false)
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, std::less<int>, std::vector<int, A1>>;
    M m      = {ar, ar + 9, A1(5)}; // implicit ctor
    assert(std::ranges::equal(m, expected));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_set(InputIterator , InputIterator, const key_compare&, const Allocator&)
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, C, std::vector<int, A1>>;
    auto m   = M(ar, ar + 9, C(3), A1(5));
    assert(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_set(InputIterator , InputIterator, const key_compare&, const Allocator&)
    // explicit(false)
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, std::less<int>, std::deque<int, A1>>;
    M m      = {ar, ar + 9, {}, A1(5)}; // implicit ctor
    assert(std::ranges::equal(m, expected));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }

  return 0;
}
