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

// template<class K> iterator       find(const K& x);
// template<class K> const_iterator find(const K& x) const;

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
concept CanFind         = requires(M m, Transparent<int> k) { m.find(k); };
using TransparentMap    = std::flat_multimap<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_multimap<int, double, NonTransparentComparator>;
static_assert(CanFind<TransparentMap>);
static_assert(CanFind<const TransparentMap>);
static_assert(!CanFind<NonTransparentMap>);
static_assert(!CanFind<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  M m            = {{"alpha", 1},
                    {"beta", 2},
                    {"beta", 0},
                    {"beta", 1},
                    {"beta", 2},
                    {"epsilon", 3},
                    {"epsilon", 1},
                    {"eta", 4},
                    {"gamma", 6},
                    {"gamma", 5}};
  const auto& cm = m;
  ASSERT_SAME_TYPE(decltype(m.find(Transparent<std::string>{"abc"})), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).find(Transparent<std::string>{"b"})), typename M::const_iterator);

  auto test_find = [&](auto&& map, const std::string& expected_key, long expected_offset) {
    auto iter = map.find(Transparent<std::string>{expected_key});
    assert(iter - map.begin() == expected_offset);
  };

  test_find(m, "alpha", 0);
  test_find(m, "beta", 1);
  test_find(m, "epsilon", 5);
  test_find(m, "eta", 7);
  test_find(m, "gamma", 8);
  test_find(m, "charlie", 10);
  test_find(m, "aaa", 10);
  test_find(m, "zzz", 10);
  test_find(cm, "alpha", 0);
  test_find(cm, "beta", 1);
  test_find(cm, "epsilon", 5);
  test_find(cm, "eta", 7);
  test_find(cm, "gamma", 8);
  test_find(cm, "charlie", 10);
  test_find(cm, "aaa", 10);
  test_find(cm, "zzz", 10);
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
    std::flat_multimap<int, int, TransparentComparator> m(std::sorted_equivalent, {{1, 1}, {2, 2}, {3, 3}, {3, 3}}, c);
    assert(!transparent_used);
    auto it = m.find(Transparent<int>{3});
    assert(it != m.end());
    assert(transparent_used);
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_multimap<std::string, int, std::less<>>;
    M m{{"alpha", 1}, {"beta", 2}, {"beta", 1}, {"eta", 3}, {"gamma", 3}};
    auto it = m.find("beta");
    assert(it == m.begin() + 1);
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
