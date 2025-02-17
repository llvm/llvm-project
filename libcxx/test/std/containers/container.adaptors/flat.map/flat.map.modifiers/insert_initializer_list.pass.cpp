//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// void insert(initializer_list<value_type> il);

#include <flat_map>
#include <cassert>
#include <functional>
#include <deque>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using V     = std::pair<const int, double>;

  M m = {{1, 1}, {1, 1.5}, {1, 2}, {3, 1}, {3, 1.5}, {3, 2}};
  m.insert({
      {4, 1},
      {4, 1.5},
      {4, 2},
      {1, 1},
      {1, 1.5},
      {1, 2},
      {2, 1},
      {2, 1.5},
      {2, 2},
  });
  assert(m.size() == 4);
  assert(std::distance(m.begin(), m.end()) == 4);
  assert(*m.begin() == V(1, 1));
  assert(*std::next(m.begin()) == V(2, 1));
  assert(*std::next(m.begin(), 2) == V(3, 1));
  assert(*std::next(m.begin(), 3) == V(4, 1));
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  {
    auto insert_func = [](auto& m, const auto& newValues) {
      using FlatMap                        = std::decay_t<decltype(m)>;
      using value_type                     = typename FlatMap::value_type;
      std::initializer_list<value_type> il = {{newValues[0].first, newValues[0].second}};
      m.insert(il);
    };
    test_insert_range_exception_guarantee(insert_func);
  }
  return 0;
}
