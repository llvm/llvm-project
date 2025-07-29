//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <list>

// void push_back(value_type&& x); // constexpr since C++26

#include <list>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::list<MoveOnly> l1;
    l1.push_back(MoveOnly(1));
    assert(l1.size() == 1);
    assert(l1.front() == MoveOnly(1));
    l1.push_back(MoveOnly(2));
    assert(l1.size() == 2);
    assert(l1.front() == MoveOnly(1));
    assert(l1.back() == MoveOnly(2));
  }
  {
    std::list<MoveOnly, min_allocator<MoveOnly>> l1;
    l1.push_back(MoveOnly(1));
    assert(l1.size() == 1);
    assert(l1.front() == MoveOnly(1));
    l1.push_back(MoveOnly(2));
    assert(l1.size() == 2);
    assert(l1.front() == MoveOnly(1));
    assert(l1.back() == MoveOnly(2));
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
