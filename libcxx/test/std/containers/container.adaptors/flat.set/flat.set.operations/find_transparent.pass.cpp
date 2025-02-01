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
#include <string>
#include <utility>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanFind         = requires(M m, Transparent<int> k) { m.find(k); };
using TransparentSet    = std::flat_set<int, TransparentComparator>;
using NonTransparentSet = std::flat_set<int, NonTransparentComparator>;
static_assert(CanFind<TransparentSet>);
static_assert(CanFind<const TransparentSet>);
static_assert(!CanFind<NonTransparentSet>);
static_assert(!CanFind<const NonTransparentSet>);

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, TransparentComparator, KeyContainer>;

  M m = {"alpha", "beta", "epsilon", "eta", "gamma"};

  const auto& cm = m;
  ASSERT_SAME_TYPE(decltype(m.find(Transparent<std::string>{"abc"})), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).find(Transparent<std::string>{"b"})), typename M::const_iterator);

  auto test_find = [&](auto&& map, const std::string& expected_key, long expected_offset) {
    auto iter = map.find(Transparent<std::string>{expected_key});
    assert(iter - map.begin() == expected_offset);
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

int main(int, char**) {
  test<std::vector<std::string>>();
  test<std::deque<std::string>>();
  test<MinSequenceContainer<std::string>>();
  test<std::vector<std::string, min_allocator<std::string>>>();

  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_set<int, TransparentComparator> m(std::sorted_unique, {1, 2, 3}, c);
    assert(!transparent_used);
    auto it = m.find(Transparent<int>{3});
    assert(it != m.end());
    assert(transparent_used);
  }

  return 0;
}
