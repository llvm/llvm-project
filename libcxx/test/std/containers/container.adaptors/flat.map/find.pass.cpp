//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

//       iterator find(const key_type& k);
// const_iterator find(const key_type& k) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
  {
    using M = std::flat_map<int, char>;
    M m = {{1,'a'}, {2,'b'}, {4,'d'}, {5,'e'}, {8,'h'}};
    ASSERT_SAME_TYPE(decltype(m.find(0)), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(0)), M::const_iterator);
    assert(m.find(0) == m.end());
    assert(m.find(1) == m.begin());
    assert(m.find(2) == m.begin() + 1);
    assert(m.find(3) == m.end());
    assert(m.find(4) == m.begin() + 2);
    assert(m.find(5) == m.begin() + 3);
    assert(m.find(6) == m.end());
    assert(m.find(7) == m.end());
    assert(std::as_const(m).find(8) == m.begin() + 4);
    assert(std::as_const(m).find(9) == m.end());
  }
  {
    using M = std::flat_map<int, char, std::greater<int>, std::deque<int, min_allocator<int>>, std::string>;
    M m = {{1,'a'}, {2,'b'}, {4,'d'}, {5,'e'}, {8,'h'}};
    ASSERT_SAME_TYPE(decltype(m.find(0)), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(0)), M::const_iterator);
    assert(m.find(0) == m.end());
    assert(m.find(1) == m.begin() + 4);
    assert(m.find(2) == m.begin() + 3);
    assert(m.find(3) == m.end());
    assert(m.find(4) == m.begin() + 2);
    assert(m.find(5) == m.begin() + 1);
    assert(m.find(6) == m.end());
    assert(m.find(7) == m.end());
    assert(std::as_const(m).find(8) == m.begin());
    assert(std::as_const(m).find(9) == m.end());
  }
  {
    using M = std::flat_map<bool, bool>;
    M m = {{true,false}, {false,true}};
    ASSERT_SAME_TYPE(decltype(m.find(0)), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(0)), M::const_iterator);
    assert(m.find(true) == m.begin() + 1);
    assert(m.find(false) == m.begin());
    m = {{true,true}};
    assert(m.find(true) == m.begin());
    assert(m.find(false) == m.end());
    m = {{false,false}};
    assert(std::as_const(m).find(true) == m.end());
    assert(std::as_const(m).find(false) == m.begin());
    m.clear();
    assert(m.find(true) == m.end());
    assert(m.find(false) == m.end());
  }
  return 0;
}
