//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> pair<iterator, bool> insert(P&& x);
// template<class K> iterator insert(const_iterator hint, P&& x);

#include <algorithm>
#include <compare>
#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <tuple>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

// Constraints: is_constructible_v<value_type, K> is true.
template <class M, class... Args>
concept CanInsert = requires(M m, Args&&... args) { m.insert(std::forward<Args>(args)...); };

using Set  = std::flat_set<int>;
using Iter = Set::const_iterator;

static_assert(CanInsert<Set, short&&>);
static_assert(CanInsert<Set, Iter, short&&>);
static_assert(!CanInsert<Set, std::string>);
static_assert(!CanInsert<Set, Iter, std::string>);

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

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;

  const int expected[] = {1, 2, 3, 4, 5};
  {
    // insert(P&&)
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    M m                                                        = {1, 2, 4, 5};
    expensive_comparisons                                      = 0;
    cheap_comparisons                                          = 0;
    std::same_as<std::pair<typename M::iterator, bool>> auto p = m.insert(3); // conversion happens first
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(p == std::make_pair(m.begin() + 2, true));
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(const_iterator, P&&)
    M m                                        = {1, 2, 4, 5};
    expensive_comparisons                      = 0;
    cheap_comparisons                          = 0;
    std::same_as<typename M::iterator> auto it = m.insert(m.begin(), 3);
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(it == m.begin() + 2);
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(value_type&&)
    M m                                                        = {1, 2, 4, 5};
    expensive_comparisons                                      = 0;
    cheap_comparisons                                          = 0;
    std::same_as<std::pair<typename M::iterator, bool>> auto p = m.insert(3); // conversion happens last
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(p == std::make_pair(m.begin() + 2, true));
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(const_iterator, value_type&&)
    M m                                        = {1, 2, 4, 5};
    expensive_comparisons                      = 0;
    cheap_comparisons                          = 0;
    std::same_as<typename M::iterator> auto it = m.insert(m.begin(), 3);
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(it == m.begin() + 2);
    assert(std::ranges::equal(m, expected));
  }
  {
    // emplace(Args&&...)
    M m                                                        = {1, 2, 4, 5};
    expensive_comparisons                                      = 0;
    cheap_comparisons                                          = 0;
    std::same_as<std::pair<typename M::iterator, bool>> auto p = m.emplace(3); // conversion happens first
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(p == std::make_pair(m.begin() + 2, true));
    assert(std::ranges::equal(m, expected));
  }
}

int main(int, char**) {
  test<std::vector<CompareCounter>>();
  test<std::deque<CompareCounter>>();
  test<MinSequenceContainer<CompareCounter>>();
  test<std::vector<CompareCounter, min_allocator<CompareCounter>>>();

  {
    // no ambiguity between insert(pos, P&&) and insert(first, last)
    using M = std::flat_set<int>;
    struct Evil {
      operator M::value_type() const;
      operator M::const_iterator() const;
    };
    std::flat_set<int> m;
    ASSERT_SAME_TYPE(decltype(m.insert(Evil())), std::pair<M::iterator, bool>);
    ASSERT_SAME_TYPE(decltype(m.insert(m.begin(), Evil())), M::iterator);
    ASSERT_SAME_TYPE(decltype(m.insert(m.begin(), m.end())), void);
  }
  {
    auto insert_func = [](auto& m, auto key_arg) {
      using FlatSet = std::decay_t<decltype(m)>;
      struct T {
        typename FlatSet::key_type key;
        T(typename FlatSet::key_type key) : key(key) {}
        operator typename FlatSet::value_type() const { return key; }
      };
      T t(key_arg);
      m.insert(t);
    };
    test_emplace_exception_guarantee(insert_func);
  }
  {
    auto insert_func_iter = [](auto& m, auto key_arg) {
      using FlatSet = std::decay_t<decltype(m)>;
      struct T {
        typename FlatSet::key_type key;
        T(typename FlatSet::key_type key) : key(key) {}
        operator typename FlatSet::value_type() const { return key; }
      };
      T t(key_arg);
      m.insert(m.begin(), t);
    };
    test_emplace_exception_guarantee(insert_func_iter);
  }
  return 0;
}
