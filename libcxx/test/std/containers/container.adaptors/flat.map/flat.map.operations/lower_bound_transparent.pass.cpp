//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> iterator       lower_bound(const K& x);
// template<class K> const_iterator lower_bound(const K& x) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <string>
#include <utility>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanLowerBound   = requires(M m, Transparent<int> k) { m.lower_bound(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanLowerBound<TransparentMap>);
static_assert(CanLowerBound<const TransparentMap>);
static_assert(!CanLowerBound<NonTransparentMap>);
static_assert(!CanLowerBound<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  M m            = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
  const auto& cm = m;
  ASSERT_SAME_TYPE(decltype(m.lower_bound(Transparent<std::string>{"abc"})), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).lower_bound(Transparent<std::string>{"b"})), typename M::const_iterator);

  auto test_lower_bound = [&](auto&& map, const std::string& expected_key, long expected_offset) {
    auto iter = map.lower_bound(Transparent<std::string>{expected_key});
    assert(iter - map.begin() == expected_offset);
  };

  test_lower_bound(m, "abc", 0);
  test_lower_bound(m, "alpha", 0);
  test_lower_bound(m, "beta", 1);
  test_lower_bound(m, "bets", 2);
  test_lower_bound(m, "charlie", 2);
  test_lower_bound(m, "echo", 2);
  test_lower_bound(m, "epsilon", 2);
  test_lower_bound(m, "eta", 3);
  test_lower_bound(m, "gamma", 4);
  test_lower_bound(m, "golf", 5);
  test_lower_bound(m, "zzz", 5);

  test_lower_bound(cm, "abc", 0);
  test_lower_bound(cm, "alpha", 0);
  test_lower_bound(cm, "beta", 1);
  test_lower_bound(cm, "bets", 2);
  test_lower_bound(cm, "charlie", 2);
  test_lower_bound(cm, "echo", 2);
  test_lower_bound(cm, "epsilon", 2);
  test_lower_bound(cm, "eta", 3);
  test_lower_bound(cm, "gamma", 4);
  test_lower_bound(cm, "golf", 5);
  test_lower_bound(cm, "zzz", 5);
}

int main(int, char**) {
  test<std::vector<std::string>, std::vector<int>>();
  test<std::deque<std::string>, std::vector<int>>();
  test<MinSequenceContainer<std::string>, MinSequenceContainer<int>>();
  test<std::vector<std::string, min_allocator<std::string>>, std::vector<int, min_allocator<int>>>();

  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto it = m.lower_bound(Transparent<int>{3});
    assert(it != m.end());
    assert(transparent_used);
  }

  return 0;
}
