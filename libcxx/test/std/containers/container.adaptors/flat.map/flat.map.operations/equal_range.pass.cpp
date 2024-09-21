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

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    using M  = std::flat_map<int, char>;
    using R  = std::pair<M::iterator, M::iterator>;
    using CR = std::pair<M::const_iterator, M::const_iterator>;
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
    using M =
        std::flat_map<int,
                      char,
                      std::greater<int>,
                      std::deque<int, min_allocator<int>>,
                      std::deque<char, min_allocator<char>>>;
    using R  = std::pair<M::iterator, M::iterator>;
    using CR = std::pair<M::const_iterator, M::const_iterator>;
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
#if 0
  // vector<bool> is not supported
  {
    using M  = std::flat_map<bool, bool>;
    using R  = std::pair<M::iterator, M::iterator>;
    using CR = std::pair<M::const_iterator, M::const_iterator>;
    M m      = {{true, false}, {false, true}};
    ASSERT_SAME_TYPE(decltype(m.equal_range(0)), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(0)), CR);
    assert(m.equal_range(true) == std::pair(m.begin() + 1, m.end()));
    assert(m.equal_range(false) == std::pair(m.begin(), m.begin() + 1));
    m = {{true, true}};
    assert(m.equal_range(true) == std::pair(m.begin(), m.end()));
    assert(m.equal_range(false) == std::pair(m.begin(), m.begin()));
    m = {{false, false}};
    assert(std::as_const(m).equal_range(true) == std::pair(m.cend(), m.cend()));
    assert(std::as_const(m).equal_range(false) == std::pair(m.cbegin(), m.cend()));
    m.clear();
    assert(m.equal_range(true) == std::pair(m.begin(), m.begin()));
    assert(m.equal_range(false) == std::pair(m.begin(), m.begin()));
  }
#endif
  return 0;
}
