//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// void insert(initializer_list<value_type> il);

#include <flat_set>
#include <cassert>
#include <functional>
#include <deque>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using M     = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using V     = typename M::value_type;

  M m = {1,1,1,3,3,3};
  m.insert({
      4,
      4,
      4,
      1,
      1,
      1,
      2,
      2,
      2,
  });
  assert(m.size() == 4);
  assert(std::distance(m.begin(), m.end()) == 4);
  assert(*m.begin() == V(1));
  assert(*std::next(m.begin()) == V(2));
  assert(*std::next(m.begin(), 2) == V(3));
  assert(*std::next(m.begin(), 3) == V(4));
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  {
    auto insert_func = [](auto& m, const auto& newValues) {
      using FlatSet                        = std::decay_t<decltype(m)>;
      using value_type                     = typename FlatSet::value_type;
      std::initializer_list<value_type> il = {newValues[0]};
      m.insert(il);
    };
    test_insert_range_exception_guarantee(insert_func);
  }
  return 0;
}
