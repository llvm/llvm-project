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
//   flat_multiset(sorted_equivalent_t, InputIterator first, InputIterator last, const key_compare& comp = key_compare());
// template<class InputIterator, class Alloc>
//   flat_multiset(sorted_equivalent_t, InputIterator first, InputIterator last, const Alloc& a);
// template<class InputIterator, class Allocator>
//   flat_multiset(sorted_equivalent_t, InputIterator first, InputIterator last, const key_compare& comp, const Allocator& a);

#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../../../test_compare.h"

void test() {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<container_type, Alloc> is true.

    using C     = test_less<int>;
    using A1    = test_allocator<int>;
    using A2    = other_allocator<int>;
    using V1    = std::vector<int, A1>;
    using V2    = std::vector<int, A2>;
    using M1    = std::flat_multiset<int, C, V1>;
    using M2    = std::flat_multiset<int, C, V2>;
    using Iter1 = typename M1::iterator;
    using Iter2 = typename M2::iterator;
    static_assert(std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const A1&>);
    static_assert(std::is_constructible_v<M2, std::sorted_equivalent_t, Iter2, Iter2, const A2&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_equivalent_t, Iter2, Iter2, const A1&>);

    static_assert(std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const C&, const A1&>);
    static_assert(std::is_constructible_v<M2, std::sorted_equivalent_t, Iter2, Iter2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_equivalent_t, Iter2, Iter2, const C&, const A1&>);
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator);
    // cpp17_input_iterator
    using M  = std::flat_multiset<int>;
    int ar[] = {1, 2, 2, 4, 5};
    auto m = M(std::sorted_equivalent, cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 5));
    auto expected = M{1, 2, 2, 4, 5};
    assert(m == expected);

    // explicit(false)
    M m2 = {std::sorted_equivalent, cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 5)};
    assert(m2 == m);
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator);
    // contiguous iterator
    using C       = test_less<int>;
    using M       = std::flat_multiset<int, C, std::vector<int, min_allocator<int>>>;
    int ar[]      = {1, 2, 4, 4, 5};
    auto m        = M(std::sorted_equivalent, ar, ar + 5);
    auto expected = M{1, 2, 4, 4, 5};
    assert(m == expected);
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&);
    // cpp_17_input_iterator
    using M  = std::flat_multiset<int, std::function<bool(int, int)>>;
    int ar[] = {1, 2, 4, 4, 5};
    auto m   = M(std::sorted_equivalent,
               cpp17_input_iterator<const int*>(ar),
               cpp17_input_iterator<const int*>(ar + 5),
               std::less<int>());
    assert(m == M({1, 2, 4, 4, 5}, std::less<>()));
    assert(m.key_comp()(1, 2) == true);

    // explicit(false)
    M m2 = {std::sorted_equivalent,
            cpp17_input_iterator<const int*>(ar),
            cpp17_input_iterator<const int*>(ar + 5),
            std::less<int>()};
    assert(m2 == m);
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&);
    // greater
    using M  = std::flat_multiset<int, std::greater<int>, std::deque<int, min_allocator<int>>>;
    int ar[] = {5, 4, 4, 2, 1};
    auto m   = M(std::sorted_equivalent,
               cpp17_input_iterator<const int*>(ar),
               cpp17_input_iterator<const int*>(ar + 5),
               std::greater<int>());
    assert((m == M{5, 4, 4, 2, 1}));
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&);
    // contiguous iterator
    using C   = test_less<int>;
    using M   = std::flat_multiset<int, C, std::vector<int, min_allocator<int>>>;
    int ar[1] = {42};
    auto m    = M(std::sorted_equivalent, ar, ar, C(5));
    assert(m.empty());
    assert(m.key_comp() == C(5));
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator , InputIterator, const Allocator&)
    using A1      = test_allocator<int>;
    using M       = std::flat_multiset<int, std::less<int>, std::vector<int, A1>>;
    int ar[]      = {1, 2, 4, 4, 5};
    auto m        = M(std::sorted_equivalent, ar, ar + 5, A1(5));
    auto expected = M{1, 2, 4, 4, 5};
    assert(m == expected);
    assert(M(m).extract().get_allocator() == A1(5));

    // explicit(false)
    M m2 = {std::sorted_equivalent, ar, ar + 5, A1(5)};
    assert(m2 == m);
    assert(std::move(m2).extract().get_allocator() == A1(5));
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&, const Allocator&);
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using M  = std::flat_multiset<int, C, std::deque<int, A1>>;
    int ar[] = {1, 2, 4, 4, 5};
    auto m   = M(std::sorted_equivalent, ar, ar + 5, C(3), A1(5));
    assert((m == M{1, 2, 4, 4, 5}));
    assert(m.key_comp() == C(3));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&, const Allocator&);
    // explicit(false)
    using A1 = test_allocator<short>;
    using M  = std::flat_multiset<short, std::less<int>, std::deque<short, A1>>;
    int ar[] = {1, 2, 4, 4, 5};
    M m      = {std::sorted_equivalent, ar, ar + 5, {}, A1(5)}; // implicit ctor
    assert((m == M{1, 2, 4, 4, 5}));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
}

int main(int, char**) {
  test();

  return 0;
}
