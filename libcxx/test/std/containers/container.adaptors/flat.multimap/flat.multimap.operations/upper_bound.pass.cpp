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

//       iterator upper_bound(const key_type& k);
// const_iterator upper_bound(const key_type& k) const;

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
    using M = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
    M m     = {
        {1, 'a'}, {2, 'b'}, {4, 'd'}, {4, 'e'}, {4, 'a'}, {4, 'b'}, {5, 'e'}, {5, 'a'}, {8, 'a'}, {8, 'b'}, {8, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.upper_bound(0)), typename M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).upper_bound(0)), typename M::const_iterator);
    assert(m.upper_bound(0) == m.begin());
    assert(m.upper_bound(1) == m.begin() + 1);
    assert(m.upper_bound(2) == m.begin() + 2);
    assert(m.upper_bound(3) == m.begin() + 2);
    assert(m.upper_bound(4) == m.begin() + 6);
    assert(m.upper_bound(5) == m.begin() + 8);
    assert(m.upper_bound(6) == m.begin() + 8);
    assert(std::as_const(m).upper_bound(7) == m.begin() + 8);
    assert(std::as_const(m).upper_bound(8) == m.end());
    assert(std::as_const(m).upper_bound(9) == m.end());
  }

  {
    using M = std::flat_multimap<Key, Value, std::greater<Key>, KeyContainer, ValueContainer>;
    M m     = {
        {1, 'a'}, {2, 'b'}, {4, 'd'}, {4, 'e'}, {4, 'a'}, {4, 'b'}, {5, 'e'}, {5, 'a'}, {8, 'a'}, {8, 'b'}, {8, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.upper_bound(0)), typename M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).upper_bound(0)), typename M::const_iterator);
    assert(m.upper_bound(0) == m.end());
    assert(m.upper_bound(1) == m.end());
    assert(m.upper_bound(2) == m.begin() + 10);
    assert(m.upper_bound(3) == m.begin() + 9);
    assert(m.upper_bound(4) == m.begin() + 9);
    assert(m.upper_bound(5) == m.begin() + 5);
    assert(m.upper_bound(6) == m.begin() + 3);
    assert(m.upper_bound(7) == m.begin() + 3);
    assert(std::as_const(m).upper_bound(8) == m.begin() + 3);
    assert(std::as_const(m).upper_bound(9) == m.begin());
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<char>>();
  test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  return 0;
}
