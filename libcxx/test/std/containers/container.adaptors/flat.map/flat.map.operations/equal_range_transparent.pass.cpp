//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> pair<iterator,iterator>             equal_range(const K& x);
// template<class K> pair<const_iterator,const_iterator> equal_range(const K& x) const;

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
concept CanEqualRange   = requires(M m, Transparent<int> k) { m.equal_range(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanEqualRange<TransparentMap>);
static_assert(CanEqualRange<const TransparentMap>);
static_assert(!CanEqualRange<NonTransparentMap>);
static_assert(!CanEqualRange<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  using R        = std::pair<typename M::iterator, typename M::iterator>;
  using CR       = std::pair<typename M::const_iterator, typename M::const_iterator>;
  M m            = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
  const auto& cm = m;
  ASSERT_SAME_TYPE(decltype(m.equal_range(Transparent<std::string>{"abc"})), R);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(Transparent<std::string>{"b"})), CR);

  auto test_found = [&](auto&& map, const std::string& expected_key, int expected_value) {
    auto [first, last] = map.equal_range(Transparent<std::string>{expected_key});
    assert(last - first == 1);
    auto [key, value] = *first;
    assert(key == expected_key);
    assert(value == expected_value);
  };

  auto test_not_found = [&](auto&& map, const std::string& expected_key, long expected_offset) {
    auto [first, last] = map.equal_range(Transparent<std::string>{expected_key});
    assert(first == last);
    assert(first - m.begin() == expected_offset);
  };

  test_found(m, "alpha", 1);
  test_found(m, "beta", 2);
  test_found(m, "epsilon", 3);
  test_found(m, "eta", 4);
  test_found(m, "gamma", 5);
  test_found(cm, "alpha", 1);
  test_found(cm, "beta", 2);
  test_found(cm, "epsilon", 3);
  test_found(cm, "eta", 4);
  test_found(cm, "gamma", 5);

  test_not_found(m, "charlie", 2);
  test_not_found(m, "aaa", 0);
  test_not_found(m, "zzz", 5);
  test_not_found(cm, "charlie", 2);
  test_not_found(cm, "aaa", 0);
  test_not_found(cm, "zzz", 5);
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
    auto p = m.equal_range(Transparent<int>{3});
    assert(p.first != p.second);
    assert(transparent_used);
  }

  return 0;
}
