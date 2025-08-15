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

// template<class K> iterator insert(P&& x);
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

using Map  = std::flat_multimap<int, double>;
using Iter = Map::const_iterator;

static_assert(CanInsert<Map, std::pair<short, double>&&>);
static_assert(CanInsert<Map, Iter, std::pair<short, double>&&>);
static_assert(CanInsert<Map, std::tuple<short, double>&&>);
static_assert(CanInsert<Map, Iter, std::tuple<short, double>&&>);
static_assert(!CanInsert<Map, int>);
static_assert(!CanInsert<Map, Iter, int>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  {
    // insert(P&&)
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    M m(std::sorted_equivalent, {{1, 1}, {2, 2}, {2, 3}, {4, 4}}, c);
    assert(!transparent_used);

    std::same_as<typename M::iterator> decltype(auto) res = m.insert(std::pair(ConvertibleTransparent<int>{3}, 3));

    assert(res->first == 3);
    assert(res->second == 3);
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    assert(!transparent_used);
  }
  {
    // insert(const_iterator, P&&)
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    M m(std::sorted_equivalent, {{1, 1}, {2, 2}, {2, 3}, {4, 4}}, c);
    std::same_as<typename M::iterator> decltype(auto) res =
        m.insert(m.begin(), std::pair(ConvertibleTransparent<int>{3}, 3));
    assert(res->first == 3);
    assert(res->second == 3);
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    assert(!transparent_used);
  }
}

constexpr bool test() {
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  {
    // no ambiguity between insert(pos, P&&) and insert(first, last)
    using M = std::flat_multimap<int, int>;
    struct Evil {
      operator M::value_type() const;
      operator M::const_iterator() const;
    };
    std::flat_multimap<int, int> m;
    ASSERT_SAME_TYPE(decltype(m.insert(Evil())), M::iterator);
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
    using M = std::flat_multimap<std::string, int, std::less<>>;
    M m{{"alpha", 1}, {"beta", 2}, {"beta", 1}, {"eta", 3}, {"gamma", 3}};
    auto it = m.insert({"beta", 1});
    assert(it == m.begin() + 3);
    auto it2 = m.insert(m.begin(), {"beta2", 2});
    assert(it2 == m.begin() + 4);
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
