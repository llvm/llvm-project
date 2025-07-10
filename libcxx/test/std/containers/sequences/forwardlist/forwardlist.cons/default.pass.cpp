//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list(); // constexpr since C++26

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef int T;
    typedef std::forward_list<T> C;
    C c;
    assert(c.empty());
  }
#if TEST_STD_VER >= 11
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    C c;
    assert(c.empty());
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    C c = {};
    assert(c.empty());
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
