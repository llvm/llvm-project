//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> size_type count(const K& x) const;

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
concept CanCount        = requires(M m, Transparent<int> k) { m.count(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanCount<TransparentMap>);
static_assert(CanCount<const TransparentMap>);
static_assert(!CanCount<NonTransparentMap>);
static_assert(!CanCount<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  M m = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
  ASSERT_SAME_TYPE(decltype(m.count(Transparent<std::string>{"abc"})), typename M::size_type);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).count(Transparent<std::string>{"b"})), typename M::size_type);
  assert(m.count(Transparent<std::string>{"alpha"}) == 1);
  assert(m.count(Transparent<std::string>{"beta"}) == 1);
  assert(m.count(Transparent<std::string>{"epsilon"}) == 1);
  assert(m.count(Transparent<std::string>{"eta"}) == 1);
  assert(m.count(Transparent<std::string>{"gamma"}) == 1);
  assert(m.count(Transparent<std::string>{"al"}) == 0);
  assert(m.count(Transparent<std::string>{""}) == 0);
  assert(m.count(Transparent<std::string>{"g"}) == 0);
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
    auto n = m.count(Transparent<int>{3});
    assert(n == 1);
    assert(transparent_used);
  }

  return 0;
}
