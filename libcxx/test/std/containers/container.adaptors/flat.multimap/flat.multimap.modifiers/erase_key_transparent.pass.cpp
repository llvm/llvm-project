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

// size_type erase(K&& k);

#include <algorithm>
#include <compare>
#include <concepts>
#include <deque>
#include <flat_map>
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
using TransparentMap    = std::flat_multimap<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_multimap<int, double, NonTransparentComparator>;
static_assert(CanErase<TransparentMap>);
static_assert(!CanErase<const TransparentMap>);
static_assert(!CanErase<NonTransparentMap>);
static_assert(!CanErase<const NonTransparentMap>);

template <class Key, class It>
struct HeterogeneousKey {
  constexpr explicit HeterogeneousKey(Key key, It it) : key_(key), it_(it) {}
  constexpr operator It() && { return it_; }
  constexpr auto operator<=>(Key key) const { return key_ <=> key; }
  constexpr friend bool operator<(const HeterogeneousKey&, const HeterogeneousKey&) {
    assert(false);
    return false;
  }
  Key key_;
  It it_;
};

template <class KeyContainer, class ValueContainer>
constexpr void test_simple() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::ranges::less, KeyContainer, ValueContainer>;

  M m = {{1, 1}, {2, 2}, {2, 2}, {3, 3}, {3, 4}, {3, 5}, {4, 4}};
  ASSERT_SAME_TYPE(decltype(m.erase(9)), typename M::size_type);
  auto n = m.erase(3); // erase(K&&) [with K=int]
  assert(n == 3);
  assert((m == M{{1, 1}, {2, 2}, {2, 2}, {4, 4}}));
  typename M::key_type lvalue = 2;
  n                           = m.erase(lvalue); // erase(K&&) [with K=int&]
  assert(n == 2);
  assert((m == M{{1, 1}, {4, 4}}));
  const typename M::key_type const_lvalue = 1;
  n                                       = m.erase(const_lvalue); // erase(const key_type&)
  assert(n == 1);
  assert((m == M{{4, 4}}));
}

template <class KeyContainer, class ValueContainer>
constexpr void test_transparent_comparator() {
  using M = std::flat_multimap<std::string, int, TransparentComparator, KeyContainer, ValueContainer>;
  using P = std::pair<std::string, int>;
  M m     = {
      {"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"epsilon", 4}, {"eta", 4}, {"gamma", 5}, {"gamma", 6}, {"gamma", 7}};
  ASSERT_SAME_TYPE(decltype(m.erase(Transparent<std::string>{"abc"})), typename M::size_type);

  auto n = m.erase(Transparent<std::string>{"epsilon"});
  assert(n == 2);
  check_invariant(m);
  assert(std::ranges::is_permutation(
      m, std::vector<P>{{"alpha", 1}, {"beta", 2}, {"eta", 4}, {"gamma", 5}, {"gamma", 6}, {"gamma", 7}}));

  auto n2 = m.erase(Transparent<std::string>{"aaa"});
  assert(n2 == 0);
  check_invariant(m);
  assert(std::ranges::is_permutation(
      m, std::vector<P>{{"alpha", 1}, {"beta", 2}, {"eta", 4}, {"gamma", 5}, {"gamma", 6}, {"gamma", 7}}));

  auto n3 = m.erase(Transparent<std::string>{"gamma"});
  assert(n3 == 3);
  assert(std::ranges::equal(m, std::vector<P>{{"alpha", 1}, {"beta", 2}, {"eta", 4}}));

  auto n4 = m.erase(Transparent<std::string>{"alpha"});
  assert(n4 == 1);
  assert(std::ranges::equal(m, std::vector<P>{{"beta", 2}, {"eta", 4}}));

  auto n5 = m.erase(Transparent<std::string>{"alpha"});
  assert(n5 == 0);
  assert(std::ranges::equal(m, std::vector<P>{{"beta", 2}, {"eta", 4}}));

  auto n6 = m.erase(Transparent<std::string>{"beta"});
  assert(n6 == 1);
  assert(std::ranges::equal(m, std::vector<P>{{"eta", 4}}));

  auto n7 = m.erase(Transparent<std::string>{"eta"});
  assert(n7 == 1);
  assert(std::ranges::equal(m, std::vector<P>{}));

  auto n8 = m.erase(Transparent<std::string>{"eta"});
  assert(n8 == 0);
  assert(std::ranges::equal(m, std::vector<P>{}));
}

constexpr bool test() {
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test_simple<std::deque<int>, std::vector<double>>();
    test_transparent_comparator<std::deque<std::string>, std::vector<int>>();
  }

  test_simple<std::vector<int>, std::vector<double>>();
  test_simple<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test_simple<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  test_transparent_comparator<std::vector<std::string>, std::vector<int>>();
  test_transparent_comparator<MinSequenceContainer<std::string>, MinSequenceContainer<int>>();
  test_transparent_comparator<std::vector<std::string, min_allocator<std::string>>,
                              std::vector<int, min_allocator<int>>>();

  {
    // P2077's HeterogeneousKey example
    using M = std::flat_multimap<int, int, std::less<>>;
    M m     = {{1, 1}, {2, 2}, {3, 3}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {6, 6}, {7, 7}, {8, 8}, {8, 8}};
    auto h1 = HeterogeneousKey<int, M::iterator>(8, m.begin());
    std::same_as<M::size_type> auto n = m.erase(h1); // lvalue is not convertible to It; erase(K&&) is the best match
    assert(n == 2);
    assert((m == M{{1, 1}, {2, 2}, {3, 3}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {6, 6}, {7, 7}}));
    std::same_as<M::iterator> auto it = m.erase(std::move(h1)); // rvalue is convertible to It; erase(K&&) drops out
    assert(it == m.begin());
    assert((m == M{{2, 2}, {3, 3}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {6, 6}, {7, 7}}));
  }
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_multimap<int, int, TransparentComparator> m(std::sorted_equivalent, {{1, 1}, {2, 2}, {3, 3}, {3, 3}}, c);
    assert(!transparent_used);
    auto n = m.erase(Transparent<int>{3});
    assert(n == 2);
    assert(transparent_used);
  }

  if (!TEST_IS_CONSTANT_EVALUATED) {
    auto erase_transparent = [](auto& m, auto key_arg) {
      using Map = std::decay_t<decltype(m)>;
      using Key = typename Map::key_type;
      m.erase(Transparent<Key>{key_arg});
    };
    test_erase_exception_guarantee(erase_transparent);
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_multimap<std::string, int, std::less<>>;
    M m{{"alpha", 1}, {"beta", 2}, {"beta", 1}, {"eta", 3}, {"gamma", 3}};
    auto n = m.erase("beta");
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
