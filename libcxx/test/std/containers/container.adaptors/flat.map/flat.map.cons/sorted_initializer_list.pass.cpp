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
//   flat_map(sorted_unique_t s, initializer_list<value_type> il,
//            const key_compare& comp = key_compare())
// template<class Alloc>
//   flat_map(sorted_unique_t, initializer_list<value_type> il, const Alloc& a);
// template<class Alloc>
//   flat_map(sorted_unique_t, initializer_list<value_type> il,
//            const key_compare& comp, const Alloc& a);

#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../../../test_compare.h"

template <class T, class U>
std::initializer_list<std::pair<T, U>> il = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};

const auto il1 = il<int, int>;
const auto il2 = il<int, short>;
const auto il3 = il<short, int>;

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
    using M1 = std::flat_map<int, int, C, V1, V1>;
    using M2 = std::flat_map<int, int, C, V1, V2>;
    using M3 = std::flat_map<int, int, C, V2, V1>;
    using IL = std::initializer_list<std::pair<int, int>>;
    static_assert(std::is_constructible_v<M1, std::sorted_unique_t, IL, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_unique_t, IL, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_unique_t, IL, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::sorted_unique_t, IL, const A2&>);

    static_assert(std::is_constructible_v<M1, std::sorted_unique_t, IL, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::sorted_unique_t, IL, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::sorted_unique_t, IL, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::sorted_unique_t, IL, const C&, const A2&>);
  }
  {
    // initializer_list<value_type> needs to match exactly
    using M = std::flat_map<int, short>;
    using C = typename M::key_compare;
    static_assert(std::is_constructible_v<M, std::sorted_unique_t, std::initializer_list<std::pair<int, short>>>);
    static_assert(std::is_constructible_v<M, std::sorted_unique_t, std::initializer_list<std::pair<int, short>>, C>);
    static_assert(std::is_constructible_v<M,
                                          std::sorted_unique_t,
                                          std::initializer_list<std::pair<int, short>>,
                                          C,
                                          std::allocator<int>>);
    static_assert(std::is_constructible_v<M,
                                          std::sorted_unique_t,
                                          std::initializer_list<std::pair<int, short>>,
                                          std::allocator<int>>);
    static_assert(
        !std::is_constructible_v<M, std::sorted_unique_t, std::initializer_list<std::pair<const int, short>>>);
    static_assert(
        !std::is_constructible_v<M, std::sorted_unique_t, std::initializer_list<std::pair<const int, short>>, C>);
    static_assert(!std::is_constructible_v<M,
                                           std::sorted_unique_t,
                                           std::initializer_list<std::pair<const int, short>>,
                                           C,
                                           std::allocator<int>>);
    static_assert(!std::is_constructible_v<M,
                                           std::sorted_unique_t,
                                           std::initializer_list<std::pair<const int, short>>,
                                           std::allocator<int>>);
    static_assert(
        !std::is_constructible_v<M, std::sorted_unique_t, std::initializer_list<std::pair<const int, const short>>>);
    static_assert(
        !std::is_constructible_v<M, std::sorted_unique_t, std::initializer_list<std::pair<const int, const short>>, C>);
    static_assert(!std::is_constructible_v<M,
                                           std::sorted_unique_t,
                                           std::initializer_list<std::pair<const int, const short>>,
                                           C,
                                           std::allocator<int>>);
    static_assert(!std::is_constructible_v<M,
                                           std::sorted_unique_t,
                                           std::initializer_list<std::pair<const int, const short>>,
                                           std::allocator<int>>);
  }

  {
    // flat_map(sorted_unique_t, initializer_list<value_type>);
    using M       = std::flat_map<int, int>;
    auto m        = M(std::sorted_unique, il1);
    auto expected = M{{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    assert(m == expected);

    // explicit(false)
    M m2 = {std::sorted_unique, il1};
    assert(m2 == m);
  }
  {
    // flat_map(sorted_unique_t, initializer_list<value_type>, const key_compare&);
    using M = std::flat_map<int, int, std::function<bool(int, int)>>;
    auto m  = M(std::sorted_unique, il1, std::less<int>());
    assert(m == M({{1, 1}, {2, 2}, {4, 4}, {5, 5}}, std::less<>()));
    assert(m.key_comp()(1, 2) == true);

    // explicit(false)
    M m2 = {std::sorted_unique, il1, std::less<int>()};
    assert(m2 == m);
  }
  {
    // flat_map(sorted_unique_t, initializer_list<value_type>, const key_compare&);
    // greater
    using M = std::flat_map<int, int, std::greater<int>, std::deque<int, min_allocator<int>>, std::vector<int>>;
    std::initializer_list<std::pair<int, int>> il4{{5, 5}, {4, 4}, {2, 2}, {1, 1}};
    auto m = M(std::sorted_unique, il4, std::greater<int>());
    assert((m == M{{5, 5}, {4, 4}, {2, 2}, {1, 1}}));
  }
  {
    // flat_map(sorted_unique_t, initializer_list<value_type>,  const Allocator&)
    using A1      = test_allocator<int>;
    using A2      = test_allocator<short>;
    using M       = std::flat_map<int, short, std::less<int>, std::vector<int, A1>, std::deque<short, A2>>;
    auto m        = M(std::sorted_unique, il2, A1(5));
    auto expected = M{{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    assert(m == expected);
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));

    // explicit(false)
    M m2 = {std::sorted_unique, il2, A1(5)};
    assert(m2 == m);
    assert(m2.keys().get_allocator() == A1(5));
    assert(m2.values().get_allocator() == A2(5));
  }
  {
    // flat_map(sorted_unique_t, initializer_list<value_type>, const key_compare&, const Allocator&);
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, C, std::vector<int, A1>, std::deque<short, A2>>;
    auto m   = M(std::sorted_unique, il2, C(3), A1(5));
    assert((m == M{{1, 1}, {2, 2}, {4, 4}, {5, 5}}));
    assert(m.key_comp() == C(3));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // flat_map(sorted_unique_t, initializer_list<value_type>, const key_compare&, const Allocator&);
    // explicit(false)
    using A1 = test_allocator<short>;
    using A2 = test_allocator<int>;
    using M  = std::flat_map<short, int, std::less<int>, std::deque<short, A1>, std::vector<int, A2>>;
    M m      = {std::sorted_unique, il3, {}, A1(5)}; // implicit ctor
    assert((m == M{{1, 1}, {2, 2}, {4, 4}, {5, 5}}));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }

  return 0;
}
