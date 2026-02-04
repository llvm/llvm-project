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
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using V   = typename M::value_type;

  {
    M m = {1, 1, 1, 3, 3, 3};
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
  {
    // was empty
    M m;
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
    M expected = {1, 2, 4};
    assert(m == expected);
  }
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
    using FlatSet                        = std::decay_t<decltype(m)>;
    using value_type                     = typename FlatSet::value_type;
    std::initializer_list<value_type> il = {newValues[0]};
    m.insert(il);
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
