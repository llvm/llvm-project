//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> iterator       find(const K& x);
// template<class K> const_iterator find(const K& x) const;

#include <cassert>
#include <flat_map>
#include <string>
#include <utility>

#include "../helpers.h"
#include "test_macros.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanFind         = requires(M m, Transparent<int> k) { m.find(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanFind<TransparentMap>);
static_assert(CanFind<const TransparentMap>);
static_assert(!CanFind<NonTransparentMap>);
static_assert(!CanFind<const NonTransparentMap>);

int main(int, char**) {
  {
    using M        = std::flat_map<std::string, int, TransparentComparator>;
    M m            = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    const auto& cm = m;
    ASSERT_SAME_TYPE(decltype(m.find(Transparent<std::string>{"abc"})), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(Transparent<std::string>{"b"})), M::const_iterator);

    auto test_find = [&](auto&& m, const std::string& expected_key, long expected_offset) {
      auto iter = m.find(Transparent<std::string>{expected_key});
      assert(iter - m.begin() == expected_offset);
    };

    test_find(m, "alpha", 0);
    test_find(m, "beta", 1);
    test_find(m, "epsilon", 2);
    test_find(m, "eta", 3);
    test_find(m, "gamma", 4);
    test_find(m, "charlie", 5);
    test_find(m, "aaa", 5);
    test_find(m, "zzz", 5);
    test_find(cm, "alpha", 0);
    test_find(cm, "beta", 1);
    test_find(cm, "epsilon", 2);
    test_find(cm, "eta", 3);
    test_find(cm, "gamma", 4);
    test_find(cm, "charlie", 5);
    test_find(cm, "aaa", 5);
    test_find(cm, "zzz", 5);
  }
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto it = m.find(Transparent<int>{3});
    assert(it != m.end());
    assert(transparent_used);
  }
#if 0
// do we really want to support this weird comparator that gives different answer for Key and Kp?
  {
    using M = std::flat_map<std::string, int, StartsWith::Less>;
    M m     = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    ASSERT_SAME_TYPE(decltype(m.find(StartsWith('b'))), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(StartsWith('b'))), M::const_iterator);
    assert(m.find("beta") == m.begin() + 1);
    assert(m.find("delta") == m.end());
    assert(m.find("zeta") == m.end());
    assert(m.find(StartsWith('b')) == m.begin() + 1);
    assert(m.find(StartsWith('d')) == m.end());
    auto it = m.find(StartsWith('e'));
    assert(m.begin() + 2 <= it && it <= m.begin() + 3); // either is acceptable
    LIBCPP_ASSERT(it == m.begin() + 2);                 // return the earliest match
    assert(m.find(StartsWith('z')) == m.end());
  }
#endif
  return 0;
}
