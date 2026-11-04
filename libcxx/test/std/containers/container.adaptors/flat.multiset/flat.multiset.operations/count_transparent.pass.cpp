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
#include <functional>
#include <string>
#include <utility>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanCount        = requires(M m, Transparent<int> k) { m.count(k); };
using TransparentSet    = std::flat_multiset<int, TransparentComparator>;
using NonTransparentSet = std::flat_multiset<int, NonTransparentComparator>;
static_assert(CanCount<TransparentSet>);
static_assert(CanCount<const TransparentSet>);
static_assert(!CanCount<NonTransparentSet>);
static_assert(!CanCount<const NonTransparentSet>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, TransparentComparator, KeyContainer>;
  {
    M m = {"alpha", "beta", "beta", "beta", "epsilon", "eta", "eta", "gamma"};
    ASSERT_SAME_TYPE(decltype(m.count(Transparent<std::string>{"abc"})), typename M::size_type);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).count(Transparent<std::string>{"b"})), typename M::size_type);
    assert(m.count(Transparent<std::string>{"alpha"}) == 1);
    assert(m.count(Transparent<std::string>{"beta"}) == 3);
    assert(m.count(Transparent<std::string>{"epsilon"}) == 1);
    assert(m.count(Transparent<std::string>{"eta"}) == 2);
    assert(m.count(Transparent<std::string>{"gamma"}) == 1);
    assert(m.count(Transparent<std::string>{"al"}) == 0);
    assert(m.count(Transparent<std::string>{""}) == 0);
    assert(m.count(Transparent<std::string>{"g"}) == 0);
  }
  {
    // empty
    M m;
    assert(m.count(Transparent<std::string>{"alpha"}) == 0);
    assert(m.count(Transparent<std::string>{"beta"}) == 0);
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
    std::flat_multiset<int, TransparentComparator> m(std::sorted_equivalent, {1, 2, 2, 2, 3, 3, 3, 3}, c);
    assert(!transparent_used);
    auto n = m.count(Transparent<int>{3});
    assert(n == 4);
    assert(transparent_used);
  }
  {
    // std::string and C string literal
    using M = std::flat_multiset<std::string, std::less<>>;
    M m     = {"alpha", "beta", "beta", "epsilon", "eta", "gamma"};
    auto n  = m.count("beta");
    assert(n == 2);
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
