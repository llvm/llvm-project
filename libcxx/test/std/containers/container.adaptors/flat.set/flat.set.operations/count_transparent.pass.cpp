//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> size_type count(const K& x) const;

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
concept CanCount        = requires(M m, Transparent<int> k) { m.count(k); };
using TransparentSet    = std::flat_set<int, TransparentComparator>;
using NonTransparentSet = std::flat_set<int, NonTransparentComparator>;
static_assert(CanCount<TransparentSet>);
static_assert(CanCount<const TransparentSet>);
static_assert(!CanCount<NonTransparentSet>);
static_assert(!CanCount<const NonTransparentSet>);

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, TransparentComparator, KeyContainer>;

  M m = {"alpha", "beta", "epsilon", "eta", "gamma"};
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
  test<std::vector<std::string>>();
  test<std::deque<std::string>>();
  test<MinSequenceContainer<std::string>>();
  test<std::vector<std::string, min_allocator<std::string>>>();

  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_set<int, TransparentComparator> m(std::sorted_unique, {1, 2, 3}, c);
    assert(!transparent_used);
    auto n = m.count(Transparent<int>{3});
    assert(n == 1);
    assert(transparent_used);
  }

  return 0;
}
