//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> pair<iterator,iterator>             equal_range(const K& x);
// template<class K> pair<const_iterator,const_iterator> equal_range(const K& x) const;

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
concept CanEqualRange   = requires(M m, Transparent<int> k) { m.equal_range(k); };
using TransparentSet    = std::flat_set<int, TransparentComparator>;
using NonTransparentSet = std::flat_set<int, NonTransparentComparator>;
static_assert(CanEqualRange<TransparentSet>);
static_assert(CanEqualRange<const TransparentSet>);
static_assert(!CanEqualRange<NonTransparentSet>);
static_assert(!CanEqualRange<const NonTransparentSet>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, TransparentComparator, KeyContainer>;

  using R  = std::pair<typename M::iterator, typename M::iterator>;
  using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;

  auto test_found = [](auto&& map, const std::string& expected_key) {
    auto [first, last] = map.equal_range(Transparent<std::string>{expected_key});
    assert(last - first == 1);
    assert(*first == expected_key);
  };

  auto test_not_found = [](auto&& map, const std::string& expected_key, long expected_offset) {
    auto [first, last] = map.equal_range(Transparent<std::string>{expected_key});
    assert(first == last);
    assert(first - map.begin() == expected_offset);
  };
  {
    M m            = {"alpha", "beta", "epsilon", "eta", "gamma"};
    const auto& cm = m;
    ASSERT_SAME_TYPE(decltype(m.equal_range(Transparent<std::string>{"abc"})), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(Transparent<std::string>{"b"})), CR);

    test_found(m, "alpha");
    test_found(m, "beta");
    test_found(m, "epsilon");
    test_found(m, "eta");
    test_found(m, "gamma");
    test_found(cm, "alpha");
    test_found(cm, "beta");
    test_found(cm, "epsilon");
    test_found(cm, "eta");
    test_found(cm, "gamma");

    test_not_found(m, "charlie", 2);
    test_not_found(m, "aaa", 0);
    test_not_found(m, "zzz", 5);
    test_not_found(cm, "charlie", 2);
    test_not_found(cm, "aaa", 0);
    test_not_found(cm, "zzz", 5);
  }
  {
    // empty
    M m;
    const auto& cm = m;
    test_not_found(m, "aaa", 0);
    test_not_found(cm, "charlie", 0);
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
    auto p = m.equal_range(Transparent<int>{3});
    assert(p.first != p.second);
    assert(transparent_used);
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_set<std::string, std::less<>>;
    M m{"alpha", "beta", "epsilon", "eta", "gamma"};
    auto [first, last] = m.equal_range("beta");
    assert(first == m.begin() + 1);
    assert(last == m.begin() + 2);
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
