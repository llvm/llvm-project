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
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  using V = std::pair<const Key, Value>;
  M m     = {{1, 1}, {1, 1.5}, {1, 2}, {3, 1}, {3, 1.5}, {3, 2}};
  m.insert(std::sorted_unique,
           {
               {0, 1},
               {1, 2},
               {2, 1},
               {4, 1},
           });
  assert(m.size() == 5);
  assert(std::distance(m.begin(), m.end()) == 5);

  assert(*m.begin() == V(0, 1));
  auto v1 = *std::next(m.begin());
  assert(v1.first == 1);
  assert(v1.second == 1 || v1.second == 1.5 || v1.second == 2);
  auto v2 = *std::next(m.begin(), 2);
  assert(v2.first == 2);
  assert(v2.second == 1);
  auto v3 = *std::next(m.begin(), 3);
  assert(v3.first == 3);
  assert(v3.second == 1 || v3.second == 1.5 || v3.second == 2);
  auto v4 = *std::next(m.begin(), 4);
  assert(v4.first == 4);
  assert(v4.second == 1);
}

constexpr bool test() {
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque<int>, std::vector<double>>();
  }
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  if (!TEST_IS_CONSTANT_EVALUATED) {
    auto insert_func = [](auto& m, const auto& newValues) {
      using FlatMap                        = std::decay_t<decltype(m)>;
      using value_type                     = typename FlatMap::value_type;
      std::initializer_list<value_type> il = {{newValues[0].first, newValues[0].second}};
      m.insert(std::sorted_unique, il);
    };
    test_insert_range_exception_guarantee(insert_func);
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
