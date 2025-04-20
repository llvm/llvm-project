//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template <class InputIterator>
//   void insert(sorted_unique_t, InputIterator first, InputIterator last);

#include <flat_map>
#include <cassert>
#include <functional>
#include <deque>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

// test constraint InputIterator
template <class M, class... Args>
concept CanInsert = requires(M m, Args&&... args) { m.insert(std::forward<Args>(args)...); };

using Map  = std::flat_map<int, int>;
using Pair = std::pair<int, int>;

static_assert(CanInsert<Map, std::sorted_unique_t, Pair*, Pair*>);
static_assert(CanInsert<Map, std::sorted_unique_t, cpp17_input_iterator<Pair*>, cpp17_input_iterator<Pair*>>);
static_assert(!CanInsert<Map, std::sorted_unique_t, int, int>);
static_assert(!CanInsert<Map, std::sorted_unique_t, cpp20_input_iterator<Pair*>, cpp20_input_iterator<Pair*>>);

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using P     = std::pair<Key, Value>;

  P ar1[] = {
      P(1, 1),
      P(2, 1),
      P(3, 1),
  };

  P ar2[] = {
      P(0, 1),
      P(2, 2),
      P(4, 1),
  };

  M m;
  m.insert(
      std::sorted_unique, cpp17_input_iterator<P*>(ar1), cpp17_input_iterator<P*>(ar1 + sizeof(ar1) / sizeof(ar1[0])));
  assert(m.size() == 3);
  M expected{{1, 1}, {2, 1}, {3, 1}};
  assert(m == expected);

  m.insert(
      std::sorted_unique, cpp17_input_iterator<P*>(ar2), cpp17_input_iterator<P*>(ar2 + sizeof(ar2) / sizeof(ar2[0])));
  assert(m.size() == 5);
  M expected2{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};
  assert(m == expected2);
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  {
    auto insert_func = [](auto& m, const auto& newValues) {
      m.insert(std::sorted_unique, newValues.begin(), newValues.end());
    };
    test_insert_range_exception_guarantee(insert_func);
  }

  return 0;
}
