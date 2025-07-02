//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// pair<iterator,iterator>             equal_range(const key_type& k);
// pair<const_iterator,const_iterator> equal_range(const key_type& k) const;

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <utility>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test_one() {
  using Key = typename KeyContainer::value_type;
  {
    using M  = std::flat_multiset<Key, std::less<>, KeyContainer>;
    using R  = std::pair<typename M::iterator, typename M::iterator>;
    using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;
    M m      = {1, 2, 2, 4, 4, 5, 5, 5, 8};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    auto begin = m.begin();
    assert(m.equal_range(0) == std::pair(begin, begin));
    assert(m.equal_range(1) == std::pair(begin, begin + 1));
    assert(m.equal_range(2) == std::pair(begin + 1, begin + 3));
    assert(m.equal_range(3) == std::pair(begin + 3, begin + 3));
    assert(m.equal_range(4) == std::pair(begin + 3, begin + 5));
    assert(m.equal_range(5) == std::pair(begin + 5, begin + 8));
    assert(m.equal_range(6) == std::pair(begin + 8, begin + 8));
    assert(m.equal_range(7) == std::pair(begin + 8, begin + 8));
    assert(std::as_const(m).equal_range(8) == std::pair(m.cbegin() + 8, m.cbegin() + 9));
    assert(std::as_const(m).equal_range(9) == std::pair(m.cbegin() + 9, m.cbegin() + 9));
  }

  {
    using M  = std::flat_multiset<Key, std::greater<int>, KeyContainer>;
    using R  = std::pair<typename M::iterator, typename M::iterator>;
    using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;
    M m      = {1, 1, 1, 2, 4, 5, 8, 8};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    auto begin = m.begin();
    assert(m.equal_range(0) == std::pair(begin + 8, begin + 8));
    assert(m.equal_range(1) == std::pair(begin + 5, begin + 8));
    assert(m.equal_range(2) == std::pair(begin + 4, begin + 5));
    assert(m.equal_range(3) == std::pair(begin + 4, begin + 4));
    assert(m.equal_range(4) == std::pair(begin + 3, begin + 4));
    assert(m.equal_range(5) == std::pair(begin + 2, begin + 3));
    assert(m.equal_range(6) == std::pair(begin + 2, begin + 2));
    assert(m.equal_range(7) == std::pair(begin + 2, begin + 2));
    assert(std::as_const(m).equal_range(8) == std::pair(m.cbegin(), m.cbegin() + 2));
    assert(std::as_const(m).equal_range(9) == std::pair(m.cbegin(), m.cbegin()));
  }
  {
    // empty
    using M = std::flat_multiset<Key, std::less<>, KeyContainer>;
    M m;
    auto end = m.end();
    assert(m.equal_range(0) == std::pair(end, end));
  }
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();
}

int main(int, char**) {
  test();

  return 0;
}
