//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> iterator       lower_bound(const K& x);
// template<class K> const_iterator lower_bound(const K& x) const;

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <string>
#include <utility>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanLowerBound   = requires(M m, Transparent<int> k) { m.lower_bound(k); };
using TransparentSet    = std::flat_multiset<int, TransparentComparator>;
using NonTransparentSet = std::flat_multiset<int, NonTransparentComparator>;
static_assert(CanLowerBound<TransparentSet>);
static_assert(CanLowerBound<const TransparentSet>);
static_assert(!CanLowerBound<NonTransparentSet>);
static_assert(!CanLowerBound<const NonTransparentSet>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, TransparentComparator, KeyContainer>;

  {
    M m            = {"alpha", "alpha", "beta", "beta", "beta", "epsilon", "eta", "eta", "eta", "eta", "gamma"};
    const auto& cm = m;
    ASSERT_SAME_TYPE(decltype(m.lower_bound(Transparent<std::string>{"abc"})), typename M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).lower_bound(Transparent<std::string>{"b"})), typename M::const_iterator);

    auto test_lower_bound = [&](auto&& set, const std::string& expected_key, long expected_offset) {
      auto iter = set.lower_bound(Transparent<std::string>{expected_key});
      assert(iter - set.begin() == expected_offset);
    };

    test_lower_bound(m, "abc", 0);
    test_lower_bound(m, "alpha", 0);
    test_lower_bound(m, "beta", 2);
    test_lower_bound(m, "bets", 5);
    test_lower_bound(m, "charlie", 5);
    test_lower_bound(m, "echo", 5);
    test_lower_bound(m, "epsilon", 5);
    test_lower_bound(m, "eta", 6);
    test_lower_bound(m, "gamma", 10);
    test_lower_bound(m, "golf", 11);
    test_lower_bound(m, "zzz", 11);

    test_lower_bound(cm, "abc", 0);
    test_lower_bound(cm, "alpha", 0);
    test_lower_bound(cm, "beta", 2);
    test_lower_bound(cm, "bets", 5);
    test_lower_bound(cm, "charlie", 5);
    test_lower_bound(cm, "echo", 5);
    test_lower_bound(cm, "epsilon", 5);
    test_lower_bound(cm, "eta", 6);
    test_lower_bound(cm, "gamma", 10);
    test_lower_bound(cm, "golf", 11);
    test_lower_bound(cm, "zzz", 11);
  }
  {
    // empty
    M m;
    auto iter = m.lower_bound(Transparent<std::string>{"a"});
    assert(iter == m.end());
  }
}

constexpr bool test() {
  test_one<std::vector<std::string>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<std::string>>();
  test_one<MinSequenceContainer<std::string>>();
  test_one<std::vector<std::string, min_allocator<std::string>>>();

  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_multiset<int, TransparentComparator> m(std::sorted_equivalent, {1, 2, 2, 3, 3}, c);
    assert(!transparent_used);
    auto it = m.lower_bound(Transparent<int>{3});
    assert(it != m.end());
    assert(transparent_used);
  }
  {
    // std::string and C string literal
    using M = std::flat_multiset<std::string, std::less<>>;
    M m     = {"alpha", "beta", "beta", "epsilon", "eta", "gamma"};
    auto it = m.lower_bound("beta");
    assert(it == m.begin() + 1);
    auto it2 = m.lower_bound("charlie");
    assert(it2 == m.begin() + 3);
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
