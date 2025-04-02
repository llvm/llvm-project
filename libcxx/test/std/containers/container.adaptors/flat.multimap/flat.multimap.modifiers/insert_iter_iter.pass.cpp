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

// template <class InputIterator>
//   void insert(InputIterator first, InputIterator last);

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

using Map  = std::flat_multimap<int, int>;
using Pair = std::pair<int, int>;

static_assert(CanInsert<Map, Pair*, Pair*>);
static_assert(CanInsert<Map, cpp17_input_iterator<Pair*>, cpp17_input_iterator<Pair*>>);
static_assert(!CanInsert<Map, int, int>);
static_assert(!CanInsert<Map, cpp20_input_iterator<Pair*>, cpp20_input_iterator<Pair*>>);

template <class KeyContainer, class ValueContainer>
void test() {
  using P = std::pair<int, double>;
  using M = std::flat_multimap<int, double, std::less<int>, KeyContainer, ValueContainer>;

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
  assert(m.size() == 9);
  std::vector<P> expected{{1, 1}, {1, 1.5}, {1, 2}, {2, 1}, {2, 1.5}, {2, 2}, {3, 1}, {3, 1.5}, {3, 2}};
  assert(std::ranges::equal(m, expected));

  m.insert(cpp17_input_iterator<P*>(ar2), cpp17_input_iterator<P*>(ar2 + sizeof(ar2) / sizeof(ar2[0])));
  assert(m.size() == 18);
  std::vector<P> expected2{
      {0, 1},
      {0, 1.5},
      {0, 2},
      {1, 1},
      {1, 1.5},
      {1, 2},
      {1, 1},
      {1, 1.5},
      {1, 2},
      {2, 1},
      {2, 1.5},
      {2, 2},
      {3, 1},
      {3, 1.5},
      {3, 2},
      {4, 1},
      {4, 1.5},
      {4, 2}};
  assert(std::ranges::equal(m, expected2));
}
int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  {
    auto insert_func = [](auto& m, const auto& newValues) { m.insert(newValues.begin(), newValues.end()); };
    test_insert_range_exception_guarantee(insert_func);
  }
  return 0;
}
