//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void resize(size_type sz, const value_type& x);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::vector<bool> v(100);
        v.resize(50, 1);
        assert(v.size() == 50);
        assert(v.capacity() >= 100);
        assert(v == std::vector<bool>(50));
        v.resize(200, 1);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        for (unsigned i = 0; i < 50; ++i)
            assert(v[i] == 0);
        for (unsigned i = 50; i < 200; ++i)
            assert(v[i] == 1);
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, min_allocator<bool>> v(100);
        v.resize(50, 1);
        assert(v.size() == 50);
        assert(v.capacity() >= 100);
        assert((v == std::vector<bool, min_allocator<bool>>(50)));
        v.resize(200, 1);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        for (unsigned i = 0; i < 50; ++i)
            assert(v[i] == 0);
        for (unsigned i = 50; i < 200; ++i)
            assert(v[i] == 1);
    }
#endif

  return 0;
}
