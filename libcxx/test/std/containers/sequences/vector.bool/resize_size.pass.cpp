//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void resize(size_type sz);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool tests()
{
    {
        std::vector<bool> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() >= 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        v.reserve(400);
        v.resize(300);  // check the case when resizing and we already have room
        assert(v.size() == 300);
        assert(v.capacity() >= 400);
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, explicit_allocator<bool>> v;
        v.resize(10);
        assert(v.size() == 10);
        assert(v.capacity() >= 10);
    }
    {
        std::vector<bool, min_allocator<bool>> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() >= 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        v.reserve(400);
        v.resize(300);  // check the case when resizing and we already have room
        assert(v.size() == 300);
        assert(v.capacity() >= 400);
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
