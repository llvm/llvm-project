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

static int expensive_comparisons = 0;
static int cheap_comparisons     = 0;

struct CompareCounter {
  int i_ = 0;
  CompareCounter(int i) : i_(i) {}
  friend auto operator<=>(const CompareCounter& x, const CompareCounter& y) {
    expensive_comparisons += 1;
    return x.i_ <=> y.i_;
  }
  bool operator==(const CompareCounter&) const = default;
  friend auto operator<=>(const CompareCounter& x, int y) {
    cheap_comparisons += 1;
    return x.i_ <=> y;
  }
};

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  {
    // insert(P&&)
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    M m                                                 = {{1, 1}, {2, 2}, {3, 1}, {3, 4}, {4, 4}, {5, 5}};
    expensive_comparisons                               = 0;
    cheap_comparisons                                   = 0;
    std::same_as<typename M::iterator> decltype(auto) r = m.insert(std::make_pair(3, 3)); // conversion happens first
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(r == m.begin() + 4);

    std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 1}, {3, 4}, {3, 3}, {4, 4}, {5, 5}};
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(const_iterator, P&&)
    M m                                        = {{1, 1}, {2, 2}, {3, 1}, {3, 4}, {4, 4}, {5, 5}};
    expensive_comparisons                      = 0;
    cheap_comparisons                          = 0;
    std::same_as<typename M::iterator> auto it = m.insert(m.begin(), std::make_pair(3, 3));
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(it == m.begin() + 2);
    std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 3}, {3, 1}, {3, 4}, {4, 4}, {5, 5}};
    assert(std::ranges::equal(m, expected));
  }
}

int main(int, char**) {
  test<std::vector<CompareCounter>, std::vector<double>>();
  test<std::deque<CompareCounter>, std::vector<double>>();
  test<MinSequenceContainer<CompareCounter>, MinSequenceContainer<double>>();
  test<std::vector<CompareCounter, min_allocator<CompareCounter>>, std::vector<double, min_allocator<double>>>();

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
  return 0;
}
