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

// bool contains(const key_type& x) const;

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
    using M = std::flat_multimap<Key, Value, std::less<>, KeyContainer, ValueContainer>;
    M m     = {{1, 1}, {2, 2}, {2, 3}, {4, 4}, {5, 5}, {8, 1}, {8, 2}, {8, 8}};
    assert(!m.contains(0));
    assert(m.contains(1));
    assert(m.contains(2));
    assert(!m.contains(3));
    assert(m.contains(4));
    assert(m.contains(5));
    assert(!m.contains(6));
    assert(!m.contains(7));
    assert(std::as_const(m).contains(8));
    assert(!std::as_const(m).contains(9));
    m.clear();
    assert(!m.contains(1));
  }
  {
    using M = std::flat_multimap<Key, Value, std::greater<int>, KeyContainer, ValueContainer>;
    M m     = {{1, 0}, {2, 0}, {4, 0}, {2, 1}, {5, 1}, {5, 2}, {5, 0}, {8, 0}};
    assert(!m.contains(0));
    assert(m.contains(1));
    assert(m.contains(2));
    assert(!m.contains(3));
    assert(m.contains(4));
    assert(m.contains(5));
    assert(!m.contains(6));
    assert(!m.contains(7));
    assert(std::as_const(m).contains(8));
    assert(!std::as_const(m).contains(9));
    m.clear();
    assert(!m.contains(1));
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<int>>();
  test<std::deque<int>, std::vector<int>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();

  return 0;
}
