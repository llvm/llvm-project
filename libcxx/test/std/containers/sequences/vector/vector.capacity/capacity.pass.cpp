//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// size_type capacity() const;

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

int main(int, char**)
{
    {
        std::vector<int> v;
        assert(v.capacity() == 0);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int> v(100);
        assert(v.capacity() == 100);
        v.push_back(0);
        assert(v.capacity() > 101);
        assert(is_contiguous_container_asan_correct(v));
    }
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> v;
        assert(v.capacity() == 0);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, min_allocator<int>> v(100);
        assert(v.capacity() == 100);
        v.push_back(0);
        assert(v.capacity() > 101);
        assert(is_contiguous_container_asan_correct(v));
    }
#endif

  return 0;
}
