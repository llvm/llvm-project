//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void flip();

#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  //
  // Testing flip() function with small vectors and various allocators
  //
  {
    std::vector<bool> v;
    v.push_back(true);
    v.push_back(false);
    v.push_back(true);
    v.flip();
    assert(!v[0]);
    assert(v[1]);
    assert(!v[2]);
  }
  {
    std::vector<bool, min_allocator<bool> > v;
    v.push_back(true);
    v.push_back(false);
    v.push_back(true);
    v.flip();
    assert(!v[0]);
    assert(v[1]);
    assert(!v[2]);
  }
  {
    std::vector<bool, test_allocator<bool> > v(test_allocator<bool>(5));
    v.push_back(true);
    v.push_back(false);
    v.push_back(true);
    v.flip();
    assert(!v[0]);
    assert(v[1]);
    assert(!v[2]);
  }

  //
  // Testing flip() function with larger vectors
  //
  {
    std::vector<bool> v(1000);
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = i & 1;
    std::vector<bool> original = v;
    v.flip();
    for (size_t i = 0; i < v.size(); ++i) {
      assert(v[i] == !original[i]);
    }
  }
  {
    std::vector<bool, min_allocator<bool> > v(1000, false, min_allocator<bool>());
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = i & 1;
    std::vector<bool, min_allocator<bool> > original = v;
    v.flip();
    for (size_t i = 0; i < v.size(); ++i)
      assert(v[i] == !original[i]);
    v.flip();
    assert(v == original);
  }
  {
    std::vector<bool, test_allocator<bool> > v(1000, false, test_allocator<bool>(5));
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = i & 1;
    std::vector<bool, test_allocator<bool> > original = v;
    v.flip();
    for (size_t i = 0; i < v.size(); ++i)
      assert(v[i] == !original[i]);
    v.flip();
    assert(v == original);
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
