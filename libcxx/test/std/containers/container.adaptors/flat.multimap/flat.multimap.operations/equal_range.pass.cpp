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
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  {
    using M  = std::flat_multimap<Key, Value, std::less<>, KeyContainer, ValueContainer>;
    using R  = std::pair<typename M::iterator, typename M::iterator>;
    using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;
    M m      = {{1, 'a'}, {1, 'a'}, {1, 'A'}, {2, 'b'}, {4, 'd'}, {5, 'E'}, {5, 'e'}, {8, 'h'}, {8, 'z'}};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    auto begin = m.begin();
    assert(m.equal_range(0) == std::pair(begin, begin));
    assert(m.equal_range(1) == std::pair(begin, begin + 3));
    assert(m.equal_range(2) == std::pair(begin + 3, begin + 4));
    assert(m.equal_range(3) == std::pair(begin + 4, begin + 4));
    assert(m.equal_range(4) == std::pair(begin + 4, begin + 5));
    assert(m.equal_range(5) == std::pair(begin + 5, begin + 7));
    assert(m.equal_range(6) == std::pair(begin + 7, begin + 7));
    assert(m.equal_range(7) == std::pair(begin + 7, begin + 7));
    assert(std::as_const(m).equal_range(8) == std::pair(m.cbegin() + 7, m.cbegin() + 9));
    assert(std::as_const(m).equal_range(9) == std::pair(m.cbegin() + 9, m.cbegin() + 9));
  }

  {
    using M  = std::flat_multimap<Key, Value, std::greater<int>, KeyContainer, ValueContainer>;
    using R  = std::pair<typename M::iterator, typename M::iterator>;
    using CR = std::pair<typename M::const_iterator, typename M::const_iterator>;
    M m      = {
        {1, 'a'}, {2, 'b'}, {2, 'b'}, {2, 'c'}, {4, 'a'}, {4, 'b'}, {4, 'c'}, {4, 'd'}, {5, 'e'}, {8, 'a'}, {8, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    auto begin = m.begin();
    assert(m.equal_range(0) == std::pair(begin + 11, begin + 11));
    assert(m.equal_range(1) == std::pair(begin + 10, begin + 11));
    assert(m.equal_range(2) == std::pair(begin + 7, begin + 10));
    assert(m.equal_range(3) == std::pair(begin + 7, begin + 7));
    assert(m.equal_range(4) == std::pair(begin + 3, begin + 7));
    assert(m.equal_range(5) == std::pair(begin + 2, begin + 3));
    assert(m.equal_range(6) == std::pair(begin + 2, begin + 2));
    assert(m.equal_range(7) == std::pair(begin + 2, begin + 2));
    assert(std::as_const(m).equal_range(8) == std::pair(m.cbegin(), m.cbegin() + 2));
    assert(std::as_const(m).equal_range(9) == std::pair(m.cbegin(), m.cbegin()));
  }
}

constexpr bool test() {
  test<std::vector<int>, std::vector<char>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
