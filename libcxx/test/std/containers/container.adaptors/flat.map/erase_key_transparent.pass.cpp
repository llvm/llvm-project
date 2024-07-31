//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// size_type erase(K&& k);

#include <compare>
#include <concepts>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>

#include "helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

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

int main(int, char**) {
  {
    using M = std::flat_map<std::string, int, StartsWith::Less>;
    M m     = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    ASSERT_SAME_TYPE(decltype(m.erase(StartsWith('b'))), M::size_type);
    M::size_type n = m.erase(StartsWith('e'));
    assert(n == 2);
    assert((m == M{{"alpha", 1}, {"beta", 2}, {"gamma", 5}}));
    n = m.erase(StartsWith('d'));
    assert(n == 0);
    assert((m == M{{"alpha", 1}, {"beta", 2}, {"gamma", 5}}));
  }
  {
    using M = std::flat_map<int, int, std::less<>>;
    M m     = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    ASSERT_SAME_TYPE(decltype(m.erase(9)), M::size_type);
    auto n = m.erase(3); // erase(K&&) [with K=int]
    assert(n == 1);
    assert((m == M{{1, 1}, {2, 2}, {4, 4}}));
    M::key_type lvalue = 2;
    n                  = m.erase(lvalue); // erase(K&&) [with K=int&]
    assert(n == 1);
    assert((m == M{{1, 1}, {4, 4}}));
    const M::key_type const_lvalue = 1;
    n                              = m.erase(const_lvalue); // erase(const key_type&)
    assert(n == 1);
    assert((m == M{{4, 4}}));
  }
  {
    using M = std::flat_map<int, int, std::less<>, std::deque<int, min_allocator<int>>, std::deque<int>>;
    M m     = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    ASSERT_SAME_TYPE(decltype(m.erase(9)), M::size_type);
    auto n = m.erase(3); // erase(K&&) [with K=int]
    assert(n == 1);
    assert((m == M{{1, 1}, {2, 2}, {4, 4}}));
    M::key_type lvalue = 2;
    n                  = m.erase(lvalue); // erase(K&&) [with K=int&]
    assert(n == 1);
    assert((m == M{{1, 1}, {4, 4}}));
    const M::key_type const_lvalue = 1;
    n                              = m.erase(const_lvalue); // erase(const key_type&)
    assert(n == 1);
    assert((m == M{{4, 4}}));
  }
  {
    // P2077's HeterogeneousKey example
    using M                           = std::flat_map<int, int, std::less<>>;
    M m                               = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
    auto h1                           = HeterogeneousKey<int, M::iterator>(8, m.begin());
    std::same_as<M::size_type> auto n = m.erase(h1); // lvalue is not convertible to It; erase(K&&) is the best match
    assert(n == 1);
    assert((m == M{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}}));
    std::same_as<M::iterator> auto it = m.erase(std::move(h1)); // rvalue is convertible to It; erase(K&&) drops out
    assert(it == m.begin());
    assert((m == M{{2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}}));
  }
  {
    using M                           = std::flat_map<int, int, std::less<>>;
    M m                               = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
    auto h1                           = HeterogeneousKey<int, M::const_iterator>(8, m.begin());
    std::same_as<M::size_type> auto n = m.erase(h1); // lvalue is not convertible to It; erase(K&&) is the best match
    assert(n == 1);
    assert((m == M{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}}));
    std::same_as<M::iterator> auto it = m.erase(std::move(h1)); // rvalue is convertible to It; erase(K&&) drops out
    assert(it == m.begin());
    assert((m == M{{2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}}));
  }
  return 0;
}
