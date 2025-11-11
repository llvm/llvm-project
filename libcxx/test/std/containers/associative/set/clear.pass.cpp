//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// constexpr void clear() noexcept; // constexpr since C++26

#include <set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::set<int> M;
    typedef int V;
    V ar[] = {1, 2, 3, 4, 5, 6, 7, 8};
    M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    assert(m.size() == 8);
    ASSERT_NOEXCEPT(m.clear());
    m.clear();
    assert(m.size() == 0);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::set<int, std::less<int>, min_allocator<int>> M;
    typedef int V;
    V ar[] = {1, 2, 3, 4, 5, 6, 7, 8};
    M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    assert(m.size() == 8);
    ASSERT_NOEXCEPT(m.clear());
    m.clear();
    assert(m.size() == 0);
  }
#endif

  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
