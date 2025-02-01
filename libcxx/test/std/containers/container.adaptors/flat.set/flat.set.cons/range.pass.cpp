//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<container-compatible-range<value_type> R>
//     flat_set(from_range_t, R&&)
// template<container-compatible-range<value_type> R>
//     flat_set(from_range_t, R&&, const key_compare&)
// template<container-compatible-range<value_type> R, class Alloc>
//      flat_set(from_range_t, R&&, const Alloc&);
// template<container-compatible-range<value_type> R, class Alloc>
//      flat_set(from_range_t, R&&, const key_compare&, const Alloc&);

#include <algorithm>
#include <deque>
#include <flat_set>
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
using Set     = std::flat_set<int>;

static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<int>>);
static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<short>>);
static_assert(!std::is_constructible_v<Set, std::from_range_t, RangeOf<std::pair<int, int>>>);

static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<int>, std::less<int>>);
static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<short>, std::less<int>>);
static_assert(!std::is_constructible_v<Set, std::from_range_t, RangeOf<std::pair<int, int>>, std::less<int>>);

static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<int>, std::allocator<int>>);
static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<short>, std::allocator<int>>);
static_assert(!std::is_constructible_v<Set, std::from_range_t, RangeOf<std::pair<int, int>>, std::allocator<int>>);

static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<int>, std::less<int>, std::allocator<int>>);
static_assert(std::is_constructible_v<Set, std::from_range_t, RangeOf<int>, std::less<int>, std::allocator<int>>);
static_assert(
    !std::
        is_constructible_v<Set, std::from_range_t, RangeOf<std::pair<int, int>>, std::less<int>, std::allocator<int>>);

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
    static_assert(std::is_constructible_v<M1, std::from_range_t, M1, const A1&>);
    static_assert(std::is_constructible_v<M2, std::from_range_t, M2, const A2&>);
    static_assert(!std::is_constructible_v<M1, std::from_range_t, M1, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::from_range_t, M2, const A1&>);

    static_assert(std::is_constructible_v<M1, std::from_range_t, M1, const C&, const A1&>);
    static_assert(std::is_constructible_v<M2, std::from_range_t, M2, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M1, std::from_range_t, M1, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, std::from_range_t, M2, const C&, const A1&>);
  }

  int ar[]       = {1, 1, 1, 2, 2, 3, 2, 3, 3};
  int expected[] = {1, 2, 3};
  {
    // flat_set(from_range_t, R&&)
    // input_range && !common
    using M    = std::flat_set<int>;
    using Iter = cpp20_input_iterator<const int*>;
    using Sent = sentinel_wrapper<Iter>;
    using R    = std::ranges::subrange<Iter, Sent>;
    auto m     = M(std::from_range, R(Iter(ar), Sent(Iter(ar + 9))));
    assert(std::ranges::equal(m, expected));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));

    // explicit(false)
    M m2 = {std::from_range, R(Iter(ar), Sent(Iter(ar + 9)))};
    assert(m2 == m);
  }
  {
    // flat_set(from_range_t, R&&)
    // greater
    using M    = std::flat_set<int, std::greater<int>, std::deque<int, min_allocator<int>>>;
    using Iter = cpp20_input_iterator<const int*>;
    using Sent = sentinel_wrapper<Iter>;
    using R    = std::ranges::subrange<Iter, Sent>;
    auto m     = M(std::from_range, R(Iter(ar), Sent(Iter(ar + 9))));
    assert(std::ranges::equal(m, std::deque<int, min_allocator<int>>{3, 2, 1}));
  }
  {
    // flat_set(from_range_t, R&&)
    // contiguous range
    using M = std::flat_set<int>;
    using R = std::ranges::subrange<const int*>;
    auto m  = M(std::from_range, R(ar, ar + 9));
    assert(std::ranges::equal(m, expected));
  }
  {
    // flat_set(from_range_t, R&&, const key_compare&)
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::vector<int>>;
    using R = std::ranges::subrange<const int*>;
    auto m  = M(std::from_range, R(ar, ar + 9), C(3));
    assert(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));

    // explicit(false)
    M m2 = {std::from_range, R(ar, ar + 9), C(3)};
    assert(m2 == m);
    assert(m2.key_comp() == C(3));
  }
  {
    // flat_set(from_range_t, R&&, const Allocator&)
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, std::less<int>, std::vector<int, A1>>;
    using R  = std::ranges::subrange<const int*>;
    auto m   = M(std::from_range, R(ar, ar + 9), A1(5));
    assert(std::ranges::equal(m, expected));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_set(from_range_t, R&&, const Allocator&)
    // explicit(false)
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, std::less<int>, std::deque<int, A1>>;
    using R  = std::ranges::subrange<const int*>;
    M m      = {std::from_range, R(ar, ar + 9), A1(5)}; // implicit ctor
    assert(std::ranges::equal(m, expected));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_set(from_range_t, R&&, const key_compare&, const Allocator&)
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, C, std::vector<int, A1>>;
    using R  = std::ranges::subrange<const int*>;
    auto m   = M(std::from_range, R(ar, ar + 9), C(3), A1(5));
    assert(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }
  {
    // flat_set(from_range_t, R&&, const key_compare&, const Allocator&)
    // explicit(false)
    using A1 = test_allocator<int>;
    using M  = std::flat_set<int, std::less<int>, std::deque<int, A1>>;
    using R  = std::ranges::subrange<const int*>;
    M m      = {std::from_range, R(ar, ar + 9), {}, A1(5)}; // implicit ctor
    assert(std::ranges::equal(m, expected));
    assert(std::move(m).extract().get_allocator() == A1(5));
  }

  return 0;
}
