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

// template<class K> bool contains(const K& x) const;

#include <cassert>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>
#include <deque>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanContains     = requires(M m, Transparent<int> k) { m.contains(k); };
using TransparentMap    = std::flat_multimap<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_multimap<int, double, NonTransparentComparator>;
static_assert(CanContains<TransparentMap>);
static_assert(CanContains<const TransparentMap>);
static_assert(!CanContains<NonTransparentMap>);
static_assert(!CanContains<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  M m = {{"alpha", 1}, {"beta", 2}, {"beta", 0}, {"epsilon", 3}, {"eta", 4}, {"eta", 1}, {"gamma", 5}};
  ASSERT_SAME_TYPE(decltype(m.contains(Transparent<std::string>{"abc"})), bool);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).contains(Transparent<std::string>{"b"})), bool);
  assert(m.contains(Transparent<std::string>{"alpha"}) == true);
  assert(m.contains(Transparent<std::string>{"beta"}) == true);
  assert(m.contains(Transparent<std::string>{"epsilon"}) == true);
  assert(m.contains(Transparent<std::string>{"eta"}) == true);
  assert(m.contains(Transparent<std::string>{"gamma"}) == true);
  assert(m.contains(Transparent<std::string>{"al"}) == false);
  assert(m.contains(Transparent<std::string>{""}) == false);
  assert(m.contains(Transparent<std::string>{"g"}) == false);
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
    std::flat_multimap<int, int, TransparentComparator> m(std::sorted_equivalent, {{1, 1}, {1, 2}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto b = m.contains(Transparent<int>{3});
    assert(b);
    assert(transparent_used);
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_multimap<std::string, int, std::less<>>;
    M m{{"alpha", 1}, {"beta", 2}, {"beta", 1}, {"eta", 3}, {"gamma", 3}};
    assert(m.contains("beta") == true);
    assert(m.contains("charlie") == false);
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
