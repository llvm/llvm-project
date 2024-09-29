//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void swap(vector& x);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    std::vector<int> v1(100);
    std::vector<int> v2(200);
    v1.swap(v2);
    assert(v1.size() == 200);
    assert(v1.capacity() == 200);
    assert(v2.size() == 100);
    assert(v2.capacity() == 100);
  }
#if TEST_STD_VER >= 11
  {
    std::vector<int, min_allocator<int>> v1(100);
    std::vector<int, min_allocator<int>> v2(200);
    v1.swap(v2);
    assert(v1.size() == 200);
    assert(v1.capacity() == 200);
    assert(v2.size() == 100);
    assert(v2.capacity() == 100);
  }
  {
    std::vector<int, safe_allocator<int>> v1(100);
    std::vector<int, safe_allocator<int>> v2(200);
    v1.swap(v2);
    assert(v1.size() == 200);
    assert(v1.capacity() == 200);
    assert(v2.size() == 100);
    assert(v2.capacity() == 100);
  }
#endif

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
