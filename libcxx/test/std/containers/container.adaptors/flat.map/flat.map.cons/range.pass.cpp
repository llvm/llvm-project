//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<container-compatible-range<value_type> R>
//     flat_map(from_range_t, R&&)
// template<container-compatible-range<value_type> R>
//     flat_map(from_range_t, R&&, const key_compare&)
// template<container-compatible-range<value_type> R, class Alloc>
//      flat_map(from_range_t, R&&, const Alloc&);
// template<container-compatible-range<value_type> R, class Alloc>
//      flat_map(from_range_t, R&&, const key_compare&, const Alloc&);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../../../test_compare.h"

// test constraint container-compatible-range

template <class V>
using RangeOf = std::ranges::subrange<V*>;
using Map     = std::flat_map<int, double>;

static_assert(std::is_constructible_v<Map, std::from_range_t, RangeOf<std::pair<int, double>>>);
static_assert(std::is_constructible_v<Map, std::from_range_t, RangeOf<std::pair<short, double>>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<double>>);

static_assert(std::is_constructible_v<Map, std::from_range_t, RangeOf<std::pair<int, double>>, std::less<int>>);
static_assert(std::is_constructible_v<Map, std::from_range_t, RangeOf<std::pair<short, double>>, std::less<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<int>, std::less<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<double>, std::less<int>>);

static_assert(std::is_constructible_v<Map, std::from_range_t, RangeOf<std::pair<int, double>>, std::allocator<int>>);
static_assert(std::is_constructible_v<Map, std::from_range_t, RangeOf<std::pair<short, double>>, std::allocator<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<int>, std::allocator<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<double>, std::allocator<int>>);

static_assert(std::is_constructible_v<Map,
                                      std::from_range_t,
                                      RangeOf<std::pair<int, double>>,
                                      std::less<int>,
                                      std::allocator<int>>);
static_assert(std::is_constructible_v<Map,
                                      std::from_range_t,
                                      RangeOf<std::pair<short, double>>,
                                      std::less<int>,
                                      std::allocator<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<int>, std::less<int>, std::allocator<int>>);
static_assert(!std::is_constructible_v<Map, std::from_range_t, RangeOf<double>, std::less<int>, std::allocator<int>>);

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
    static_assert(std::is_constructible_v<M1, std::from_range_t, M1, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::from_range_t, M1, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::from_range_t, M2, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::from_range_t, M3, const A2&>);

    static_assert(std::is_constructible_v<M1, std::from_range_t, M1, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, std::from_range_t, M1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::from_range_t, M2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, std::from_range_t, M3, const C&, const A2&>);
  }
  {
    // container-compatible-range
    using C           = test_less<int>;
    using A1          = test_allocator<int>;
    using A2          = test_allocator<std::string>;
    using M           = std::flat_map<int, std::string, C, std::vector<int, A1>, std::vector<std::string, A2>>;
    using Pair        = std::pair<int, std::string>;
    using PairLike    = std::tuple<int, std::string>;
    using NonPairLike = int;

    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<Pair>&>);
    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<PairLike>&>);
    static_assert(!std::is_constructible_v<M, std::from_range_t, std::vector<NonPairLike>&>);

    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<Pair>&, const C&>);
    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<PairLike>&, const C&>);
    static_assert(!std::is_constructible_v<M, std::from_range_t, std::vector<NonPairLike>&, const C&>);

    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<Pair>&, const A1&>);
    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<PairLike>&, const A1&>);
    static_assert(!std::is_constructible_v<M, std::from_range_t, std::vector<NonPairLike>&, const A1&>);

    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<Pair>&, const C&, const A1&>);
    static_assert(std::is_constructible_v<M, std::from_range_t, std::vector<PairLike>&, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M, std::from_range_t, std::vector<NonPairLike>&, const C&, const A1&>);
  }

  using P      = std::pair<int, short>;
  P ar[]       = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {2, 7}, {3, 8}, {3, 9}};
  P expected[] = {{1, 1}, {2, 4}, {3, 6}};
  {
    // flat_map(from_range_t, R&&)
    // input_range && !common
    using M    = std::flat_map<int, short>;
    using Iter = cpp20_input_iterator<const P*>;
    using Sent = sentinel_wrapper<Iter>;
    using R    = std::ranges::subrange<Iter, Sent>;
    auto m     = M(std::from_range, R(Iter(ar), Sent(Iter(ar + 9))));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));

    // explicit(false)
    M m2 = {std::from_range, R(Iter(ar), Sent(Iter(ar + 9)))};
    assert(m2 == m);
  }
  {
    // flat_map(from_range_t, R&&)
    // greater
    using M    = std::flat_map<int, short, std::greater<int>, std::deque<int, min_allocator<int>>, std::deque<short>>;
    using Iter = cpp20_input_iterator<const P*>;
    using Sent = sentinel_wrapper<Iter>;
    using R    = std::ranges::subrange<Iter, Sent>;
    auto m     = M(std::from_range, R(Iter(ar), Sent(Iter(ar + 9))));
    assert((m.keys() == std::deque<int, min_allocator<int>>{3, 2, 1}));
    LIBCPP_ASSERT((m.values() == std::deque<short>{6, 4, 1}));
  }
  {
    // flat_map(from_range_t, R&&)
    // contiguous range
    using M = std::flat_map<int, short>;
    using R = std::ranges::subrange<const P*>;
    auto m  = M(std::from_range, R(ar, ar + 9));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
  }
  {
    // flat_map(from_range_t, R&&, const key_compare&)
    using C = test_less<int>;
    using M = std::flat_map<int, short, C, std::vector<int>, std::deque<short>>;
    using R = std::ranges::subrange<const P*>;
    auto m  = M(std::from_range, R(ar, ar + 9), C(3));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));

    // explicit(false)
    M m2 = {std::from_range, R(ar, ar + 9), C(3)};
    assert(m2 == m);
    assert(m2.key_comp() == C(3));
  }
  {
    // flat_map(from_range_t, R&&, const Allocator&)
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, std::less<int>, std::vector<int, A1>, std::deque<short, A2>>;
    using R  = std::ranges::subrange<const P*>;
    auto m   = M(std::from_range, R(ar, ar + 9), A1(5));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // flat_map(from_range_t, R&&, const Allocator&)
    // explicit(false)
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, std::less<int>, std::vector<int, A1>, std::deque<short, A2>>;
    using R  = std::ranges::subrange<const P*>;
    M m      = {std::from_range, R(ar, ar + 9), A1(5)}; // implicit ctor
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // flat_map(from_range_t, R&&, const key_compare&, const Allocator&)
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, C, std::vector<int, A1>, std::deque<short, A2>>;
    using R  = std::ranges::subrange<const P*>;
    auto m   = M(std::from_range, R(ar, ar + 9), C(3), A1(5));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    // flat_map(from_range_t, R&&, const key_compare&, const Allocator&)
    // explicit(false)
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, std::less<int>, std::deque<int, A1>, std::vector<short, A2>>;
    using R  = std::ranges::subrange<const P*>;
    M m      = {std::from_range, R(ar, ar + 9), {}, A2(5)}; // implicit ctor
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }

  return 0;
}
