//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_multiset& operator=(initializer_list<value_type> il);

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <ranges>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

template <class KeyContainer>
constexpr void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  {
    M m = {8, 10};
    assert(m.size() == 2);
    m              = {3, 1, 2, 2, 3, 4, 3, 5, 6, 5};
    int expected[] = {1, 2, 2, 3, 3, 3, 4, 5, 5, 6};
    assert(std::ranges::equal(m, expected));
  }
  {
    M m = {10, 8};
    assert(m.size() == 2);
    m                 = {3};
    double expected[] = {3};
    assert(std::ranges::equal(m, expected));
  }
  {
    // was empty
    M m;
    assert(m.size() == 0);
    m              = {3, 1, 2, 2, 3, 4, 3, 5, 6, 5};
    int expected[] = {1, 2, 2, 3, 3, 3, 4, 5, 5, 6};
    assert(std::ranges::equal(m, expected));
  }
}

constexpr bool test() {
  test<std::vector<int>>();
  test<std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
