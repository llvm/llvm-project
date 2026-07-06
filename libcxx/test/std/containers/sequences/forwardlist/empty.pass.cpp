//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// class forward_list

// bool empty() const noexcept; // constexpr since C++26

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::forward_list<int> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert(c.empty());
    c.push_front(C::value_type(1));
    assert(!c.empty());
    c.clear();
    assert(c.empty());
  }
#if TEST_STD_VER >= 11
  {
    typedef std::forward_list<int, min_allocator<int>> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert(c.empty());
    c.push_front(C::value_type(1));
    assert(!c.empty());
    c.clear();
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
