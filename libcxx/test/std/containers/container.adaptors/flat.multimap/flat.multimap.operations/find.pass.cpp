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

//       iterator find(const key_type& k);
// const_iterator find(const key_type& k) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  M m = {{1, 'a'}, {1, 'a'}, {1, 'b'}, {2, 'c'}, {2, 'b'}, {4, 'a'}, {4, 'd'}, {5, 'e'}, {8, 'a'}, {8, 'h'}};
  ASSERT_SAME_TYPE(decltype(m.find(0)), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(std::as_const(m).find(0)), typename M::const_iterator);
  assert(m.find(0) == m.end());
  assert(m.find(1) == m.begin());
  assert(m.find(2) == m.begin() + 3);
  assert(m.find(3) == m.end());
  assert(m.find(4) == m.begin() + 5);
  assert(m.find(5) == m.begin() + 7);
  assert(m.find(6) == m.end());
  assert(m.find(7) == m.end());
  assert(std::as_const(m).find(8) == m.begin() + 8);
  assert(std::as_const(m).find(9) == m.end());
}

int main(int, char**) {
  test<std::vector<int>, std::vector<char>>();
  test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  return 0;
}
