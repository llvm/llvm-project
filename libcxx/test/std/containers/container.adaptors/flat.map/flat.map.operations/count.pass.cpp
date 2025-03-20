//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// size_type count(const key_type& x) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <utility>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;

  {
    using M = std::flat_map<Key, Value, std::less<>, KeyContainer, ValueContainer>;
    M m     = {{1, 1}, {2, 2}, {4, 4}, {5, 5}, {8, 8}};
    ASSERT_SAME_TYPE(decltype(m.count(0)), size_t);
    assert(m.count(0) == 0);
    assert(m.count(1) == 1);
    assert(m.count(2) == 1);
    assert(m.count(3) == 0);
    assert(m.count(4) == 1);
    assert(m.count(5) == 1);
    assert(m.count(6) == 0);
    assert(m.count(7) == 0);
    assert(std::as_const(m).count(8) == 1);
    assert(std::as_const(m).count(9) == 0);
  }
  {
    using M = std::flat_map<Key, Value, std::greater<int>, KeyContainer, ValueContainer>;
    M m     = {{1, 0}, {2, 0}, {4, 0}, {5, 0}, {8, 0}};
    ASSERT_SAME_TYPE(decltype(m.count(0)), size_t);
    assert(m.count(0) == 0);
    assert(m.count(1) == 1);
    assert(m.count(2) == 1);
    assert(m.count(3) == 0);
    assert(m.count(4) == 1);
    assert(m.count(5) == 1);
    assert(m.count(6) == 0);
    assert(m.count(7) == 0);
    assert(std::as_const(m).count(8) == 1);
    assert(std::as_const(m).count(9) == 0);
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<int>>();
  test<std::deque<int>, std::vector<int>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();

  return 0;
}
