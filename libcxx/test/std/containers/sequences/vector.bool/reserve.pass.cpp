//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void reserve(size_type n);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::vector<bool> v;
        v.reserve(10);
        assert(v.capacity() >= 10);
    }
    {
        std::vector<bool> v(100);
        assert(v.capacity() >= 100);
        v.reserve(50);
        assert(v.size() == 100);
        assert(v.capacity() >= 100);
        v.reserve(150);
        assert(v.size() == 100);
        assert(v.capacity() >= 150);
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, min_allocator<bool>> v;
        v.reserve(10);
        assert(v.capacity() >= 10);
    }
    {
        std::vector<bool, min_allocator<bool>> v(100);
        assert(v.capacity() >= 100);
        v.reserve(50);
        assert(v.size() == 100);
        assert(v.capacity() >= 100);
        v.reserve(150);
        assert(v.size() == 100);
        assert(v.capacity() >= 150);
    }
#endif

  return 0;
}
