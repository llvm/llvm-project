//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template <class InputIterator>
//   void insert(sorted_unique_t, InputIterator first, InputIterator last);

#include <flat_set>
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

using Set = std::flat_set<int>;

static_assert(CanInsert<Set, std::sorted_unique_t, int*, int*>);
static_assert(CanInsert<Set, std::sorted_unique_t, cpp17_input_iterator<int*>, cpp17_input_iterator<int*>>);
static_assert(!CanInsert<Set, std::sorted_unique_t, int, int>);
static_assert(!CanInsert<Set, std::sorted_unique_t, cpp20_input_iterator<int*>, cpp20_input_iterator<int*>>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;

  int ar1[] = {1, 2, 3};

  int ar2[] = {0, 2, 4};

  M m;
  m.insert(std::sorted_unique,
           cpp17_input_iterator<int*>(ar1),
           cpp17_input_iterator<int*>(ar1 + sizeof(ar1) / sizeof(ar1[0])));
  assert(m.size() == 3);
  M expected{1, 2, 3};
  assert(m == expected);

  m.insert(std::sorted_unique,
           cpp17_input_iterator<int*>(ar2),
           cpp17_input_iterator<int*>(ar2 + sizeof(ar2) / sizeof(ar2[0])));
  assert(m.size() == 5);
  M expected2{0, 1, 2, 3, 4};
  assert(m == expected2);
}

constexpr bool test() {
  test_one<std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  return true;
}

void test_exception() {
  auto insert_func = [](auto& m, const auto& newValues) {
    m.insert(std::sorted_unique, newValues.begin(), newValues.end());
  };
  test_insert_range_exception_guarantee(insert_func);
}

int main(int, char**) {
  test();
  test_exception();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
