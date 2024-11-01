//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void resize(size_type sz);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "asan_testing.h"

TEST_CONSTEXPR_CXX20 bool tests() {
    {
        std::vector<int> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        // Add 1 for implementations that dynamically allocate a container proxy.
        std::vector<int, limited_allocator<int, 300 + 1> > v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
#if TEST_STD_VER >= 11
    {
        std::vector<MoveOnly> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        // Add 1 for implementations that dynamically allocate a container proxy.
        std::vector<MoveOnly, limited_allocator<MoveOnly, 300 + 1> > v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<MoveOnly, min_allocator<MoveOnly>> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
      std::vector<int, safe_allocator<int>> v(100);
      v.resize(50);
      assert(v.size() == 50);
      assert(v.capacity() == 100);
      assert(is_contiguous_container_asan_correct(v));
      v.resize(200);
      assert(v.size() == 200);
      assert(v.capacity() >= 200);
      assert(is_contiguous_container_asan_correct(v));
    }
#endif

    return true;
}

int main(int, char**)
{
    tests();

#if TEST_STD_VER > 17
    static_assert(tests());
#endif

  return 0;
}
