//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set& operator=(initializer_list<value_type> il);

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  {
    M m = {8, 10};
    assert(m.size() == 2);
    m              = {3, 1, 2, 2, 3, 4, 3, 5, 6, 5};
    int expected[] = {1, 2, 3, 4, 5, 6};
    assert(std::ranges::equal(m, expected));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
  }
  {
    M m = {10, 8};
    assert(m.size() == 2);
    m              = {3};
    int expected[] = {3};
    assert(std::ranges::equal(m, expected));
  }
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();
  test<std::vector<int, min_allocator<int>>>();

  return 0;
}
