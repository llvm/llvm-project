//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

//       reverse_iterator rbegin();
// const_reverse_iterator rbegin() const;
//       reverse_iterator rend();
// const_reverse_iterator rend()   const;
//
// const_reverse_iterator crbegin() const;
// const_reverse_iterator crend()   const;

#include <cassert>
#include <cstddef>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
  {
    using M = std::flat_map<int, char, std::less<int>, std::deque<int>, std::deque<char>>;
    M m = {{1,'a'}, {2,'b'}, {3,'c'}, {4,'d'}};
    ASSERT_SAME_TYPE(decltype(m.rbegin()), M::reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.crbegin()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.rend()), M::reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.crend()), M::const_reverse_iterator);
    assert(m.size() == 4);
    assert(std::distance(m.rbegin(), m.rend()) == 4);
    assert(std::distance(m.crbegin(), m.crend()) == 4);
    M::reverse_iterator i;  // default-construct
    ASSERT_SAME_TYPE(decltype(i->first), const int&);
    ASSERT_SAME_TYPE(decltype(i->second), char&);
    i = m.rbegin();  // move-assignment
    M::const_reverse_iterator k = i;  // converting constructor
    assert(i == k);  // comparison
    for (int j = 4; j >= 1; --j, ++i) {  // pre-increment
      assert(i->first == j);  // operator->
      assert(i->second == 'a' + j - 1);
    }
    assert(i == m.rend());
    for (int j = 1; j <= 4; ++j) {
      --i;  // pre-decrement
      assert((*i).first == j);
      assert((*i).second == 'a' + j - 1);
    }
    assert(i == m.rbegin());
  }
  {
    using M = std::flat_map<short, char, std::less<>, std::deque<short>, std::string>;
    const M m = {{1,'a'}, {2,'b'}, {3,'c'}, {4,'d'}};
    ASSERT_SAME_TYPE(decltype(m.rbegin()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.crbegin()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.rend()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.crend()), M::const_reverse_iterator);
    assert(m.size() == 4);
    assert(std::distance(m.rbegin(), m.rend()) == 4);
    assert(std::distance(m.crbegin(), m.crend()) == 4);
    M::const_reverse_iterator i;  // default-construct 
    ASSERT_SAME_TYPE(decltype(i->first), const short&);
    ASSERT_SAME_TYPE(decltype(i->second), const char&);
    i = m.rbegin();  // move-assignment
    for (int j = 4; j >= 1; --j, ++i) {  // pre-increment
      assert(i->first == j);
      assert(i->second == 'a' + j - 1);
    }
    assert(i == m.rend());
    for (int j = 1; j <= 4; ++j) {
      --i;  // pre-decrement
      assert((*i).first == j);
      assert((*i).second == 'a' + j - 1);
    }
    assert(i == m.rbegin());
  }
  {
    // N3644 testing
    using C = std::flat_map<int, char>;
    C::reverse_iterator ii1{}, ii2{};
    C::reverse_iterator ii4 = ii1;
    C::const_reverse_iterator cii{};
    assert(ii1 == ii2);
    assert(ii1 == ii4);
    assert(!(ii1 != ii2));

    assert( (ii1 == cii));
    assert( (cii == ii1));
    assert(!(ii1 != cii));
    assert(!(cii != ii1));
  }
  return 0;
}
