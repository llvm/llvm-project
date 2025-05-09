//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// void insert(sorted_equivalent_t, initializer_list<value_type> il);

#include <flat_set>
#include <cassert>
#include <functional>
#include <deque>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  {
    M m = {1, 1, 1, 3, 3, 3};
    m.insert(std::sorted_equivalent, {0, 1, 1, 2, 2, 4});
    assert(m.size() == 12);
    M expected = {0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4};
    assert(m == expected);
  }
  {
    // empty
    M m;
    m.insert(std::sorted_equivalent, {0, 1, 1, 2, 2, 4});
    M expected = {0, 1, 1, 2, 2, 4};
    assert(m == expected);
  }
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();
}

void test_exception() {
  auto insert_func = [](auto& m, const auto& newValues) {
    using FlatSet                        = std::decay_t<decltype(m)>;
    using value_type                     = typename FlatSet::value_type;
    std::initializer_list<value_type> il = {newValues[0]};
    m.insert(std::sorted_equivalent, il);
  };
  test_insert_range_exception_guarantee(insert_func);
}

int main(int, char**) {
  test();
  test_exception();

  return 0;
}
