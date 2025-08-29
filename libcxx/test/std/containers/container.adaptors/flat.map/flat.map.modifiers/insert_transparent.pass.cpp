//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> pair<iterator, bool> insert(P&& x);
// template<class K> iterator insert(const_iterator hint, P&& x);

#include <algorithm>
#include <compare>
#include <concepts>
#include <deque>
#include <flat_map>
#include <functional>
#include <tuple>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

// Constraints: is_constructible_v<pair<key_type, mapped_type>, P> is true.
template <class M, class... Args>
concept CanInsert = requires(M m, Args&&... args) { m.insert(std::forward<Args>(args)...); };

using Map  = std::flat_map<int, double>;
using Iter = Map::const_iterator;

static_assert(CanInsert<Map, std::pair<short, double>&&>);
static_assert(CanInsert<Map, Iter, std::pair<short, double>&&>);
static_assert(CanInsert<Map, std::tuple<short, double>&&>);
static_assert(CanInsert<Map, Iter, std::tuple<short, double>&&>);
static_assert(!CanInsert<Map, int>);
static_assert(!CanInsert<Map, Iter, int>);

constexpr bool test() {
  {
    // template<class K> pair<iterator, bool> insert(P&& x);
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    using M = std::flat_map<int, int, TransparentComparator>;
    M m(std::sorted_unique, {{1, 1}, {2, 2}, {4, 4}}, c);
    assert(!transparent_used);

    std::same_as<std::pair<typename M::iterator, bool>> decltype(auto) res =
        m.insert(std::pair(ConvertibleTransparent<int>{3}, 3));

    assert(res.second);
    assert(res.first->first == 3);
    assert(res.first->second == 3);
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    assert(!transparent_used);
  }
  {
    // template<class K> iterator insert(const_iterator hint, P&& x);
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    using M = std::flat_map<int, int, TransparentComparator>;
    M m(std::sorted_unique, {{1, 1}, {2, 2}, {4, 4}}, c);
    assert(!transparent_used);

    std::same_as<typename M::iterator> decltype(auto) res =
        m.insert(m.begin(), std::pair(ConvertibleTransparent<int>{3}, 3));

    assert(res->first == 3);
    assert(res->second == 3);
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    assert(!transparent_used);
  }
  {
    // no ambiguity between insert(pos, P&&) and insert(first, last)
    using M = std::flat_map<int, int>;
    struct Evil {
      operator M::value_type() const;
      operator M::const_iterator() const;
    };
    std::flat_map<int, int> m;
    ASSERT_SAME_TYPE(decltype(m.insert(Evil())), std::pair<M::iterator, bool>);
    ASSERT_SAME_TYPE(decltype(m.insert(m.begin(), Evil())), M::iterator);
    ASSERT_SAME_TYPE(decltype(m.insert(m.begin(), m.end())), void);
  }

  if (!TEST_IS_CONSTANT_EVALUATED) {
    {
      auto insert_func = [](auto& m, auto key_arg, auto value_arg) {
        using FlatMap    = std::decay_t<decltype(m)>;
        using tuple_type = std::tuple<typename FlatMap::key_type, typename FlatMap::mapped_type>;
        tuple_type t(key_arg, value_arg);
        m.insert(t);
      };
      test_emplace_exception_guarantee(insert_func);
    }
    {
      auto insert_func_iter = [](auto& m, auto key_arg, auto value_arg) {
        using FlatMap    = std::decay_t<decltype(m)>;
        using tuple_type = std::tuple<typename FlatMap::key_type, typename FlatMap::mapped_type>;
        tuple_type t(key_arg, value_arg);
        m.insert(m.begin(), t);
      };
      test_emplace_exception_guarantee(insert_func_iter);
    }
  }
  {
    // LWG4239 std::string and C string literal
    using M = std::flat_map<std::string, int, std::less<>>;
    M m{{"alpha", 1}, {"beta", 2}, {"epsilon", 1}, {"eta", 3}, {"gamma", 3}};
    auto [it, inserted] = m.insert({"alpha", 1});
    assert(!inserted);
    assert(it == m.begin());
    auto it2 = m.insert(m.begin(), {"beta2", 2});
    assert(it2 == m.begin() + 2);
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
