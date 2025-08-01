//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> bool contains(const K& x) const;

#include <cassert>
#include <flat_set>
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
using TransparentSet    = std::flat_set<int, TransparentComparator>;
using NonTransparentSet = std::flat_set<int, NonTransparentComparator>;
static_assert(CanContains<TransparentSet>);
static_assert(CanContains<const TransparentSet>);
static_assert(!CanContains<NonTransparentSet>);
static_assert(!CanContains<const NonTransparentSet>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, TransparentComparator, KeyContainer>;

  {
    M m = {"alpha", "beta", "epsilon", "eta", "gamma"};
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
  {
    // empty
    M m;
    assert(m.contains(Transparent<std::string>{"gamma"}) == false);
    assert(m.contains(Transparent<std::string>{"al"}) == false);
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
    std::flat_set<int, TransparentComparator> m(std::sorted_unique, {1, 2, 3}, c);
    assert(!transparent_used);
    auto b = m.contains(Transparent<int>{3});
    assert(b);
    assert(transparent_used);
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_set<std::string, std::less<>>;
    M m{"alpha", "beta", "epsilon", "eta", "gamma"};
    assert(m.contains("beta"));
    assert(!m.contains("eta2"));
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
