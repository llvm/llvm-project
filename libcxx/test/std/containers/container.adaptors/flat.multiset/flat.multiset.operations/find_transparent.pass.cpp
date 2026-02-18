//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> iterator       find(const K& x);
// template<class K> const_iterator find(const K& x) const;

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
concept CanFind         = requires(M m, Transparent<int> k) { m.find(k); };
using TransparentSet    = std::flat_multiset<int, TransparentComparator>;
using NonTransparentSet = std::flat_multiset<int, NonTransparentComparator>;
static_assert(CanFind<TransparentSet>);
static_assert(CanFind<const TransparentSet>);
static_assert(!CanFind<NonTransparentSet>);
static_assert(!CanFind<const NonTransparentSet>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, TransparentComparator, KeyContainer>;

  {
    M m = {"alpha", "alpha", "alpha", "beta", "epsilon", "epsilon", "eta", "gamma", "gamma"};

    const auto& cm = m;
    ASSERT_SAME_TYPE(decltype(m.find(Transparent<std::string>{"abc"})), typename M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(Transparent<std::string>{"b"})), typename M::const_iterator);

    auto test_find = [](auto&& set, const std::string& expected_key, long expected_offset) {
      auto iter = set.find(Transparent<std::string>{expected_key});
      assert(iter - set.begin() == expected_offset);
    };

    test_find(m, "alpha", 0);
    test_find(m, "beta", 3);
    test_find(m, "epsilon", 4);
    test_find(m, "eta", 6);
    test_find(m, "gamma", 7);
    test_find(m, "charlie", 9);
    test_find(m, "aaa", 9);
    test_find(m, "zzz", 9);
    test_find(cm, "alpha", 0);
    test_find(cm, "beta", 3);
    test_find(cm, "epsilon", 4);
    test_find(cm, "eta", 6);
    test_find(cm, "gamma", 7);
    test_find(cm, "charlie", 9);
    test_find(cm, "aaa", 9);
    test_find(cm, "zzz", 9);
  }
  {
    // empty
    M m;
    auto iter = m.find(Transparent<std::string>{"a"});
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
    std::flat_multiset<int, TransparentComparator> m(std::sorted_equivalent, {1, 2, 2, 2, 3, 3}, c);
    assert(!transparent_used);
    auto it = m.find(Transparent<int>{3});
    assert(it != m.end());
    assert(transparent_used);
  }
  {
    // std::string and C string literal
    using M = std::flat_multiset<std::string, std::less<>>;
    M m     = {"alpha", "beta", "beta", "epsilon", "eta", "gamma"};
    auto it = m.find("beta");
    assert(it == m.begin() + 1);
    auto it2 = m.find("charlie");
    assert(it2 == m.end());
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
