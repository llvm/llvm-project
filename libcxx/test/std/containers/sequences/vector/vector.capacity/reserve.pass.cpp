//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void reserve(size_type n);

#include <vector>
#include <cassert>
#include <stdexcept>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    std::vector<int> v;
    v.reserve(10);
    assert(v.capacity() >= 10);
    assert(is_contiguous_container_asan_correct(v));
  }
  {
    std::vector<int> v(100);
    assert(v.capacity() == 100);
    v.reserve(50);
    assert(v.size() == 100);
    assert(v.capacity() == 100);
    v.reserve(150);
    assert(v.size() == 100);
    assert(v.capacity() == 150);
    assert(is_contiguous_container_asan_correct(v));
  }
  {
    // Add 1 for implementations that dynamically allocate a container proxy.
    std::vector<int, limited_allocator<int, 250 + 1> > v(100);
    assert(v.capacity() == 100);
    v.reserve(50);
    assert(v.size() == 100);
    assert(v.capacity() == 100);
    v.reserve(150);
    assert(v.size() == 100);
    assert(v.capacity() == 150);
    assert(is_contiguous_container_asan_correct(v));
  }
#if TEST_STD_VER >= 11
  {
    std::vector<int, min_allocator<int>> v;
    v.reserve(10);
    assert(v.capacity() >= 10);
    assert(is_contiguous_container_asan_correct(v));
  }
  {
    std::vector<int, min_allocator<int>> v(100);
    assert(v.capacity() == 100);
    v.reserve(50);
    assert(v.size() == 100);
    assert(v.capacity() == 100);
    v.reserve(150);
    assert(v.size() == 100);
    assert(v.capacity() == 150);
    assert(is_contiguous_container_asan_correct(v));
  }
  {
    std::vector<int, safe_allocator<int>> v;
    v.reserve(10);
    assert(v.capacity() >= 10);
    assert(is_contiguous_container_asan_correct(v));
  }
  {
    std::vector<int, safe_allocator<int>> v(100);
    assert(v.capacity() == 100);
    v.reserve(50);
    assert(v.size() == 100);
    assert(v.capacity() == 100);
    v.reserve(150);
    assert(v.size() == 100);
    assert(v.capacity() == 150);
    assert(is_contiguous_container_asan_correct(v));
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
