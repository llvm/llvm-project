//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// explicit list(const Alloc& = Alloc()); // constexpr since C++26

#include <list>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::list<int> l;
    assert(l.size() == 0);
    assert(std::distance(l.begin(), l.end()) == 0);
  }
  {
    std::list<int> l((std::allocator<int>()));
    assert(l.size() == 0);
    assert(std::distance(l.begin(), l.end()) == 0);
  }
  {
    std::list<int, limited_allocator<int, 4> > l;
    assert(l.size() == 0);
    assert(std::distance(l.begin(), l.end()) == 0);
  }
#if TEST_STD_VER >= 11
  {
    std::list<int, min_allocator<int>> l;
    assert(l.size() == 0);
    assert(std::distance(l.begin(), l.end()) == 0);
  }
  {
    std::list<int, min_allocator<int>> l((min_allocator<int>()));
    assert(l.size() == 0);
    assert(std::distance(l.begin(), l.end()) == 0);
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
