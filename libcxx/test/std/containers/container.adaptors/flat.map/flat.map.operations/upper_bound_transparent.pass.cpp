//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> iterator       upper_bound(const K& x);
// template<class K> const_iterator upper_bound(const K& x) const;

#include <cassert>
#include <flat_map>
#include <string>
#include <utility>

#include "../helpers.h"
#include "test_macros.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanUpperBound   = requires(M m, Transparent<int> k) { m.upper_bound(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanUpperBound<TransparentMap>);
static_assert(CanUpperBound<const TransparentMap>);
static_assert(!CanUpperBound<NonTransparentMap>);
static_assert(!CanUpperBound<const NonTransparentMap>);

int main(int, char**) {
  {
    using M        = std::flat_map<std::string, int, TransparentComparator>;
    M m            = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    const auto& cm = m;
    ASSERT_SAME_TYPE(decltype(m.lower_bound(Transparent<std::string>{"abc"})), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).lower_bound(Transparent<std::string>{"b"})), M::const_iterator);

    auto test_upper_bound = [&](auto&& m, const std::string& expected_key, long expected_offset) {
      auto iter = m.upper_bound(Transparent<std::string>{expected_key});
      assert(iter - m.begin() == expected_offset);
    };

    test_upper_bound(m, "abc", 0);
    test_upper_bound(m, "alpha", 1);
    test_upper_bound(m, "beta", 2);
    test_upper_bound(m, "bets", 2);
    test_upper_bound(m, "charlie", 2);
    test_upper_bound(m, "echo", 2);
    test_upper_bound(m, "epsilon", 3);
    test_upper_bound(m, "eta", 4);
    test_upper_bound(m, "gamma", 5);
    test_upper_bound(m, "golf", 5);
    test_upper_bound(m, "zzz", 5);

    test_upper_bound(cm, "abc", 0);
    test_upper_bound(cm, "alpha", 1);
    test_upper_bound(cm, "beta", 2);
    test_upper_bound(cm, "bets", 2);
    test_upper_bound(cm, "charlie", 2);
    test_upper_bound(cm, "echo", 2);
    test_upper_bound(cm, "epsilon", 3);
    test_upper_bound(cm, "eta", 4);
    test_upper_bound(cm, "gamma", 5);
    test_upper_bound(cm, "golf", 5);
    test_upper_bound(cm, "zzz", 5);
  }
#if 0
// do we really want to support this weird comparator that gives different answer for Key and Kp?
  {
    using M = std::flat_map<std::string, int, StartsWith::Less>;
    M m     = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    ASSERT_SAME_TYPE(decltype(m.upper_bound(StartsWith('b'))), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).upper_bound(StartsWith('b'))), M::const_iterator);
    assert(m.upper_bound("beta") == m.begin() + 2);
    assert(m.upper_bound("delta") == m.begin() + 2);
    assert(m.upper_bound("zeta") == m.begin() + 5);
    assert(m.upper_bound(StartsWith('b')) == m.begin() + 2);
    assert(m.upper_bound(StartsWith('d')) == m.begin() + 2);
    assert(m.upper_bound(StartsWith('e')) == m.begin() + 4);
    assert(m.upper_bound(StartsWith('z')) == m.begin() + 5);
  }
#endif
  return 0;
}
