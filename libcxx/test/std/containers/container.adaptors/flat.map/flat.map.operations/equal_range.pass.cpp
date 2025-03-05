//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// pair<iterator,iterator>             equal_range(const key_type& k);
// pair<const_iterator,const_iterator> equal_range(const key_type& k) const;

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
    using M  = std::flat_map<Key, Value, std::less<>, KeyContainer, ValueContainer>;
    using R  = std::pair<typename M::iterator, typename M::iterator>;
    using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;
    M m      = {{1, 'a'}, {2, 'b'}, {4, 'd'}, {5, 'e'}, {8, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    auto begin = m.begin();
    assert(m.equal_range(0) == std::pair(begin, begin));
    assert(m.equal_range(1) == std::pair(begin, begin + 1));
    assert(m.equal_range(2) == std::pair(begin + 1, begin + 2));
    assert(m.equal_range(3) == std::pair(begin + 2, begin + 2));
    assert(m.equal_range(4) == std::pair(begin + 2, begin + 3));
    assert(m.equal_range(5) == std::pair(begin + 3, begin + 4));
    assert(m.equal_range(6) == std::pair(begin + 4, begin + 4));
    assert(m.equal_range(7) == std::pair(begin + 4, begin + 4));
    assert(std::as_const(m).equal_range(8) == std::pair(m.cbegin() + 4, m.cbegin() + 5));
    assert(std::as_const(m).equal_range(9) == std::pair(m.cbegin() + 5, m.cbegin() + 5));
  }

  {
    using M  = std::flat_map<Key, Value, std::greater<int>, KeyContainer, ValueContainer>;
    using R  = std::pair<typename M::iterator, typename M::iterator>;
    using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;
    M m      = {{1, 'a'}, {2, 'b'}, {4, 'd'}, {5, 'e'}, {8, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    auto begin = m.begin();
    assert(m.equal_range(0) == std::pair(begin + 5, begin + 5));
    assert(m.equal_range(1) == std::pair(begin + 4, begin + 5));
    assert(m.equal_range(2) == std::pair(begin + 3, begin + 4));
    assert(m.equal_range(3) == std::pair(begin + 3, begin + 3));
    assert(m.equal_range(4) == std::pair(begin + 2, begin + 3));
    assert(m.equal_range(5) == std::pair(begin + 1, begin + 2));
    assert(m.equal_range(6) == std::pair(begin + 1, begin + 1));
    assert(m.equal_range(7) == std::pair(begin + 1, begin + 1));
    assert(std::as_const(m).equal_range(8) == std::pair(m.cbegin(), m.cbegin() + 1));
    assert(std::as_const(m).equal_range(9) == std::pair(m.cbegin(), m.cbegin()));
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<char>>();
  test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  return 0;
}
