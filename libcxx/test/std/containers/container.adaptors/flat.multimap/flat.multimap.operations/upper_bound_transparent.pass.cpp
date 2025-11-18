//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// template<class K> iterator       upper_bound(const K& x);
// template<class K> const_iterator upper_bound(const K& x) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanUpperBound   = requires(M m, Transparent<int> k) { m.upper_bound(k); };
using TransparentMap    = std::flat_multimap<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_multimap<int, double, NonTransparentComparator>;
static_assert(CanUpperBound<TransparentMap>);
static_assert(CanUpperBound<const TransparentMap>);
static_assert(!CanUpperBound<NonTransparentMap>);
static_assert(!CanUpperBound<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  M m            = {{"alpha", 1},
                    {"alpha", 2},
                    {"alpha", 3},
                    {"beta", 2},
                    {"epsilon", 3},
                    {"epsilon", 4},
                    {"eta", 4},
                    {"gamma", 5},
                    {"gamma", 5},
                    {"gamma", 5},
                    {"gamma", 5}};
  const auto& cm = m;
  ASSERT_SAME_TYPE(decltype(m.lower_bound(Transparent<std::string>{"abc"})), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).lower_bound(Transparent<std::string>{"b"})), typename M::const_iterator);

  auto test_upper_bound = [&](auto&& map, const std::string& expected_key, long expected_offset) {
    auto iter = map.upper_bound(Transparent<std::string>{expected_key});
    assert(iter - map.begin() == expected_offset);
  };

  test_upper_bound(m, "abc", 0);
  test_upper_bound(m, "alpha", 3);
  test_upper_bound(m, "beta", 4);
  test_upper_bound(m, "bets", 4);
  test_upper_bound(m, "charlie", 4);
  test_upper_bound(m, "echo", 4);
  test_upper_bound(m, "epsilon", 6);
  test_upper_bound(m, "eta", 7);
  test_upper_bound(m, "gamma", 11);
  test_upper_bound(m, "golf", 11);
  test_upper_bound(m, "zzz", 11);

  test_upper_bound(cm, "abc", 0);
  test_upper_bound(cm, "alpha", 3);
  test_upper_bound(cm, "beta", 4);
  test_upper_bound(cm, "bets", 4);
  test_upper_bound(cm, "charlie", 4);
  test_upper_bound(cm, "echo", 4);
  test_upper_bound(cm, "epsilon", 6);
  test_upper_bound(cm, "eta", 7);
  test_upper_bound(cm, "gamma", 11);
  test_upper_bound(cm, "golf", 11);
  test_upper_bound(cm, "zzz", 11);
}

constexpr bool test() {
  test<std::vector<std::string>, std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<std::string>, std::vector<int>>();
  test<MinSequenceContainer<std::string>, MinSequenceContainer<int>>();
  test<std::vector<std::string, min_allocator<std::string>>, std::vector<int, min_allocator<int>>>();
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_multimap<int, int, TransparentComparator> m(std::sorted_equivalent, {{1, 1}, {2, 2}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto it = m.upper_bound(Transparent<int>{2});
    assert(it == m.begin() + 3);
    assert(transparent_used);
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_multimap<std::string, int, std::less<>>;
    M m{{"alpha", 1}, {"beta", 2}, {"beta", 1}, {"eta", 3}, {"gamma", 3}};
    auto it = m.upper_bound("charlie");
    assert(it == m.begin() + 3);
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
