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
#include <flat_map>
#include <functional>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

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

int main(int, char**) {
  const std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
  {
    // insert(P&&)
    //   Unlike flat_set, here we can't use key_compare to compare value_type versus P,
    //   so we must eagerly convert to value_type.
    using M                                           = std::flat_map<CompareCounter, int, std::less<>>;
    M m                                               = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    expensive_comparisons                             = 0;
    cheap_comparisons                                 = 0;
    std::same_as<std::pair<M::iterator, bool>> auto p = m.insert(std::make_pair(3, 3)); // conversion happens first
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(p == std::make_pair(m.begin() + 2, true));
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(const_iterator, P&&)
    using M                           = std::flat_map<CompareCounter, int, std::less<>>;
    M m                               = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    expensive_comparisons             = 0;
    cheap_comparisons                 = 0;
    std::same_as<M::iterator> auto it = m.insert(m.begin(), std::make_pair(3, 3));
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(it == m.begin() + 2);
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(value_type&&)
    using M                                           = std::flat_map<CompareCounter, int>;
    M m                                               = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    expensive_comparisons                             = 0;
    cheap_comparisons                                 = 0;
    std::same_as<std::pair<M::iterator, bool>> auto p = m.insert(std::make_pair(3, 3)); // conversion happens last
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(p == std::make_pair(m.begin() + 2, true));
    assert(std::ranges::equal(m, expected));
  }
  {
    // insert(const_iterator, value_type&&)
    using M                           = std::flat_map<CompareCounter, int>;
    M m                               = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    expensive_comparisons             = 0;
    cheap_comparisons                 = 0;
    std::same_as<M::iterator> auto it = m.insert(m.begin(), std::make_pair(3, 3));
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(it == m.begin() + 2);
    assert(std::ranges::equal(m, expected));
  }
  {
    // emplace(Args&&...)
    using M                                           = std::flat_map<CompareCounter, int>;
    M m                                               = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    expensive_comparisons                             = 0;
    cheap_comparisons                                 = 0;
    std::same_as<std::pair<M::iterator, bool>> auto p = m.emplace(std::make_pair(3, 3)); // conversion happens first
    assert(expensive_comparisons >= 2);
    assert(cheap_comparisons == 0);
    assert(p == std::make_pair(m.begin() + 2, true));
    assert(std::ranges::equal(m, expected));
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
  return 0;
}
