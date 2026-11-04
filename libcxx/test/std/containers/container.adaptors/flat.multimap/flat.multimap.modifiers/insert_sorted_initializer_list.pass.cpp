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

// void insert(sorted_equivalent_t, initializer_list<value_type> il);

#include <algorithm>
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
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  using V = std::pair<const Key, Value>;
  M m     = {{1, 1}, {1, 1.5}, {1, 2}, {3, 1}, {3, 1.5}, {3, 2}};
  m.insert(std::sorted_equivalent,
           {
               {0, 1},
               {1, 2},
               {1, 3},
               {2, 1},
               {2, 4},
               {4, 1},
           });
  assert(m.size() == 12);
  V expected[] = {{0, 1}, {1, 1}, {1, 1.5}, {1, 2}, {1, 2}, {1, 3}, {2, 1}, {2, 4}, {3, 1}, {3, 1.5}, {3, 2}, {4, 1}};
  assert(std::ranges::is_permutation(m, expected));
  check_invariant(m);
}

constexpr bool test() {
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  if (!TEST_IS_CONSTANT_EVALUATED) {
    auto insert_func = [](auto& m, const auto& newValues) {
      using FlatMap                        = std::decay_t<decltype(m)>;
      using value_type                     = typename FlatMap::value_type;
      std::initializer_list<value_type> il = {{newValues[0].first, newValues[0].second}};
      m.insert(std::sorted_equivalent, il);
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
