//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// class list

// allocator_type get_allocator() const // constexpr since C++26

#include <list>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::allocator<int> alloc;
    const std::list<int> l(alloc);
    assert(l.get_allocator() == alloc);
  }
  {
    other_allocator<int> alloc(1);
    const std::list<int, other_allocator<int> > l(alloc);
    assert(l.get_allocator() == alloc);
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
