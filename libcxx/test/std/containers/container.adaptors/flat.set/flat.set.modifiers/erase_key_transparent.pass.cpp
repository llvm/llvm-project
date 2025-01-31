//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// size_type erase(K&& k);

#include <compare>
#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanErase        = requires(M m, Transparent<int> k) { m.erase(k); };
using TransparentSet    = std::flat_set<int, TransparentComparator>;
using NonTransparentSet = std::flat_set<int, NonTransparentComparator>;
static_assert(CanErase<TransparentSet>);
static_assert(!CanErase<const TransparentSet>);
static_assert(!CanErase<NonTransparentSet>);
static_assert(!CanErase<const NonTransparentSet>);

template <class Key, class It>
struct HeterogeneousKey {
  explicit HeterogeneousKey(Key key, It it) : key_(key), it_(it) {}
  operator It() && { return it_; }
  auto operator<=>(Key key) const { return key_ <=> key; }
  friend bool operator<(const HeterogeneousKey&, const HeterogeneousKey&) {
    assert(false);
    return false;
  }
  Key key_;
  It it_;
};

template <class KeyContainer>
void test_simple() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;

  M m = {1, 2, 3, 4};
  ASSERT_SAME_TYPE(decltype(m.erase(9)), typename M::size_type);
  auto n = m.erase(3); // erase(K&&) [with K=int]
  assert(n == 1);
  assert((m == M{1, 2, 4}));
  typename M::key_type lvalue = 2;
  n                           = m.erase(lvalue); // erase(K&&) [with K=int&]
  assert(n == 1);
  assert((m == M{1, 4}));
  const typename M::key_type const_lvalue = 1;
  n                                       = m.erase(const_lvalue); // erase(const key_type&)
  assert(n == 1);
  assert((m == M{4}));
}

template <class KeyContainer>
void test_transparent_comparator() {
  using M = std::flat_set<std::string, TransparentComparator, KeyContainer>;
  M m     = {"alpha", "beta", "epsilon", "eta", "gamma"};
  ASSERT_SAME_TYPE(decltype(m.erase(Transparent<std::string>{"abc"})), typename M::size_type);

  auto n = m.erase(Transparent<std::string>{"epsilon"});
  assert(n == 1);

  M expected = {"alpha", "beta", "eta", "gamma"};
  assert(m == expected);

  auto n2 = m.erase(Transparent<std::string>{"aaa"});
  assert(n2 == 0);
  assert(m == expected);
}

int main(int, char**) {
  test_simple<std::vector<int>>();
  test_simple<std::deque<int>>();
  test_simple<MinSequenceContainer<int>>();
  test_simple<std::vector<int, min_allocator<int>>>();

  test_transparent_comparator<std::vector<std::string>>();
  test_transparent_comparator<std::deque<std::string>>();
  test_transparent_comparator<MinSequenceContainer<std::string>>();
  test_transparent_comparator<std::vector<std::string, min_allocator<std::string>>>();

  {
    // P2077's HeterogeneousKey example
    using M                           = std::flat_set<int, std::less<>>;
    M m                               = {1, 2, 3, 4, 5, 6, 7, 8};
    auto h1                           = HeterogeneousKey<int, M::iterator>(8, m.begin());
    std::same_as<M::size_type> auto n = m.erase(h1); // lvalue is not convertible to It; erase(K&&) is the best match
    assert(n == 1);
    assert((m == M{1, 2, 3, 4, 5, 6, 7}));
    std::same_as<M::iterator> auto it = m.erase(std::move(h1)); // rvalue is convertible to It; erase(K&&) drops out
    assert(it == m.begin());
    assert((m == M{2, 3, 4, 5, 6, 7}));
  }
  {
    using M                           = std::flat_set<int, std::less<>>;
    M m                               = {1, 2, 3, 4, 5, 6, 7, 8};
    auto h1                           = HeterogeneousKey<int, M::const_iterator>(8, m.begin());
    std::same_as<M::size_type> auto n = m.erase(h1); // lvalue is not convertible to It; erase(K&&) is the best match
    assert(n == 1);
    assert((m == M{1, 2, 3, 4, 5, 6, 7}));
    std::same_as<M::iterator> auto it = m.erase(std::move(h1)); // rvalue is convertible to It; erase(K&&) drops out
    assert(it == m.begin());
    assert((m == M{2, 3, 4, 5, 6, 7}));
  }
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_set<int, TransparentComparator> m(std::sorted_unique, {1, 2, 3}, c);
    assert(!transparent_used);
    auto n = m.erase(Transparent<int>{3});
    assert(n == 1);
    assert(transparent_used);
  }
  {
    auto erase_transparent = [](auto& m, auto key_arg) {
      using Set = std::decay_t<decltype(m)>;
      using Key = typename Set::key_type;
      m.erase(Transparent<Key>{key_arg});
    };
    test_erase_exception_guarantee(erase_transparent);
  }

  return 0;
}
