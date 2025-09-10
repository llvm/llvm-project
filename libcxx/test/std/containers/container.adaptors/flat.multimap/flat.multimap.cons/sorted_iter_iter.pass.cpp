//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template <class InputIterator>
//   flat_multimap(sorted_equivalent_t, InputIterator first, InputIterator last, const key_compare& comp = key_compare());
// template<class InputIterator, class Alloc>
//   flat_multimap(InputIterator first, InputIterator last, const Alloc& a);
// template<class InputIterator, class Allocator>
//   flat_multimap(sorted_equivalent_t, InputIterator first, InputIterator last, const key_compare& comp, const Allocator& a);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "MinSequenceContainer.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../../../test_compare.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;

  {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator);
    // cpp17_input_iterator
    using M = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
    using P = std::pair<Key, Value>;
    P ar[]  = {{1, 1}, {4, 2}, {4, 4}, {5, 5}};
    auto m  = M(std::sorted_equivalent, cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 4));
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {4, 2}, {4, 4}, {5, 5}}));

    // explicit(false)
    M m2 = {std::sorted_equivalent, cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 4)};
    assert(m2 == m);
  }
  {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator);
    // contiguous iterator
    using C = test_less<Key>;
    using P = std::pair<Key, Value>;
    using M = std::flat_multimap<Key, Value, C, KeyContainer, ValueContainer>;
    P ar[]  = {{1, 1}, {4, 2}, {4, 4}, {5, 5}};
    auto m  = M(std::sorted_equivalent, ar, ar + 4);
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {4, 2}, {4, 4}, {5, 5}}));
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&);
    // cpp_17_input_iterator
    using M = std::flat_multimap<Key, Value, std::function<bool(Key, Value)>>;
    using P = std::pair<Key, Value>;
    P ar[]  = {{1, 1}, {4, 2}, {4, 4}, {5, 5}};
    auto m  = M(std::sorted_equivalent,
               cpp17_input_iterator<const P*>(ar),
               cpp17_input_iterator<const P*>(ar + 4),
               std::less<int>());
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {4, 2}, {4, 4}, {5, 5}}));
    assert(m.key_comp()(1, 2) == true);

    // explicit(false)
    M m2 = {std::sorted_equivalent,
            cpp17_input_iterator<const P*>(ar),
            cpp17_input_iterator<const P*>(ar + 4),
            std::less<int>()};
    assert(m2 == m);
  }
  {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&);
    // greater
    using M = std::flat_multimap<Key, Value, std::greater<int>, KeyContainer, ValueContainer>;
    using P = std::pair<Key, Value>;
    P ar[]  = {{5, 5}, {4, 4}, {4, 2}, {1, 1}};
    auto m  = M(std::sorted_equivalent,
               cpp17_input_iterator<const P*>(ar),
               cpp17_input_iterator<const P*>(ar + 4),
               std::greater<int>());
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{5, 5}, {4, 4}, {4, 2}, {1, 1}}));
  }
  {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&);
    // contiguous iterator
    using C                     = test_less<Key>;
    using M                     = std::flat_multimap<Key, Value, C, KeyContainer, ValueContainer>;
    std::pair<Key, Value> ar[1] = {{42, 42}};
    auto m                      = M(std::sorted_equivalent, ar, ar, C(5));
    assert(m.empty());
    assert(m.key_comp() == C(5));
  }
}

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test_alloc() {
  {
    // flat_multimap(sorted_equivalent_t, InputIterator , InputIterator, const Allocator&)
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_multimap<int, short, std::less<int>, KeyContainer<int, A1>, ValueContainer<short, A2>>;
    using P  = std::pair<int, int>;
    P ar[]   = {{1, 1}, {4, 2}, {4, 4}, {5, 5}};
    auto m   = M(std::sorted_equivalent, ar, ar + 4, A1(5));
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {4, 2}, {4, 4}, {5, 5}}));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));

    // explicit(false)
    M m2 = {std::sorted_equivalent, ar, ar + 4, A1(5)};
    assert(m2 == m);
    assert(m2.keys().get_allocator() == A1(5));
    assert(m2.values().get_allocator() == A2(5));
  }
  {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&, const Allocator&);
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_multimap<int, short, C, KeyContainer<int, A1>, ValueContainer<short, A2>>;
    using P  = std::pair<int, int>;
    P ar[]   = {{1, 1}, {4, 2}, {4, 4}, {5, 5}};
    auto m   = M(std::sorted_equivalent, ar, ar + 4, C(3), A1(5));
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {4, 2}, {4, 4}, {5, 5}}));
    assert(m.key_comp() == C(3));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator, const key_compare&, const Allocator&);
    // explicit(false)
    using A1 = test_allocator<short>;
    using A2 = test_allocator<int>;
    using M  = std::flat_multimap<short, int, std::less<int>, KeyContainer<short, A1>, ValueContainer<int, A2>>;
    using P  = std::pair<int, int>;
    P ar[]   = {{1, 1}, {4, 2}, {4, 4}, {5, 5}};
    M m      = {std::sorted_equivalent, ar, ar + 4, {}, A1(5)}; // implicit ctor
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {4, 2}, {4, 4}, {5, 5}}));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
}

constexpr bool test() {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<key_container_type, Alloc> is true
    // and uses_allocator_v<mapped_container_type, Alloc> is true.
    using C     = test_less<int>;
    using A1    = test_allocator<int>;
    using A2    = other_allocator<int>;
    using V1    = std::vector<int, A1>;
    using V2    = std::vector<int, A2>;
    using M1    = std::flat_multimap<int, int, C, V1, V1>;
    using M2    = std::flat_multimap<int, int, C, V1, V2>;
    using M3    = std::flat_multimap<int, int, C, V2, V1>;
    using Iter1 = typename M1::iterator;
    using Iter2 = typename M2::iterator;
    using Iter3 = typename M3::iterator;
    static_assert(std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_equivalent_t, Iter2, Iter2, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::sorted_equivalent_t, Iter3, Iter3, const A2&>);

    static_assert(std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_equivalent_t, Iter1, Iter1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_equivalent_t, Iter2, Iter2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::sorted_equivalent_t, Iter3, Iter3, const C&, const A2&>);
  }

  test<std::vector<int>, std::vector<int>>();
  test<std::vector<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();

  test_alloc<std::vector, std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque<int>, std::vector<double>>();
    test_alloc<std::deque, std::deque>();
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
