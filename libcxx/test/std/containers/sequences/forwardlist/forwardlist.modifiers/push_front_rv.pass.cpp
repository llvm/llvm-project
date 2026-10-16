//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// void push_front(value_type&& v); // constexpr since C++26

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef MoveOnly T;
    typedef std::forward_list<T> C;
    C c;
    c.push_front(1);
    assert(c.front() == 1);
    assert(std::distance(c.begin(), c.end()) == 1);
    c.push_front(3);
    assert(c.front() == 3);
    assert(*std::next(c.begin()) == 1);
    assert(std::distance(c.begin(), c.end()) == 2);
  }
  {
    typedef MoveOnly T;
    typedef std::forward_list<T, min_allocator<T>> C;
    C c;
    c.push_front(1);
    assert(c.front() == 1);
    assert(std::distance(c.begin(), c.end()) == 1);
    c.push_front(3);
    assert(c.front() == 3);
    assert(*std::next(c.begin()) == 1);
    assert(std::distance(c.begin(), c.end()) == 2);
  }

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
