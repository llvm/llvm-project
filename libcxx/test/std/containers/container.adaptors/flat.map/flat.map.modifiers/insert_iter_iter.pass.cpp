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
//   void insert(InputIterator first, InputIterator last);

#include <algorithm>
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

static_assert(CanInsert<Map, Pair*, Pair*>);
static_assert(CanInsert<Map, cpp17_input_iterator<Pair*>, cpp17_input_iterator<Pair*>>);
static_assert(!CanInsert<Map, int, int>);
static_assert(!CanInsert<Map, cpp20_input_iterator<Pair*>, cpp20_input_iterator<Pair*>>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using P = std::pair<int, double>;
  using M = std::flat_map<int, double, std::less<int>, KeyContainer, ValueContainer>;

  P ar1[] = {
      P(2, 1),
      P(2, 1.5),
      P(2, 2),
      P(1, 1),
      P(1, 1.5),
      P(1, 2),
      P(3, 1),
      P(3, 1.5),
      P(3, 2),
  };
  P ar2[] = {
      P(4, 1),
      P(4, 1.5),
      P(4, 2),
      P(1, 1),
      P(1, 1.5),
      P(1, 2),
      P(0, 1),
      P(0, 1.5),
      P(0, 2),
  };

  M m;
  m.insert(cpp17_input_iterator<P*>(ar1), cpp17_input_iterator<P*>(ar1 + sizeof(ar1) / sizeof(ar1[0])));
  assert(m.size() == 3);

  assert(std::ranges::equal(m.keys(), KeyContainer{1, 2, 3}));
  check_possible_values(
      m.values(),
      std::vector<std::vector<double>>{
          {1, 1.5, 2},
          {1, 1.5, 2},
          {1, 1.5, 2},
      });

  auto m2 = m;

  m2.insert(cpp17_input_iterator<P*>(ar2), cpp17_input_iterator<P*>(ar2 + sizeof(ar2) / sizeof(ar2[0])));
  assert(m2.size() == 5);

  assert(std::ranges::equal(m2.keys(), KeyContainer{0, 1, 2, 3, 4}));
  check_possible_values(
      m2.values(),
      std::vector<std::vector<double>>{
          {1, 1.5, 2},
          {m[1]},
          {m[2]},
          {m[3]},
          {1, 1.5, 2},
      });
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
    auto insert_func = [](auto& m, const auto& newValues) { m.insert(newValues.begin(), newValues.end()); };
    test_insert_range_exception_guarantee(insert_func);
  }
  {
    std::flat_map<int, int, std::less<int>, SillyReserveVector<int>, SillyReserveVector<int>> m{{1, 1}, {2, 2}};
    std::vector<std::pair<int, int>> v{{3, 3}, {4, 4}};
    m.insert(v.begin(), v.end());
    assert(std::ranges::equal(m, std::vector<std::pair<int, int>>{{1, 1}, {2, 2}, {3, 3}, {4, 4}}));
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
