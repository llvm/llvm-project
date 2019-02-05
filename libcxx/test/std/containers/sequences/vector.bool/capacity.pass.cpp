//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// size_type capacity() const;

#include <vector>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
        std::vector<bool> v;
        assert(v.capacity() == 0);
    }
    {
        std::vector<bool> v(100);
        assert(v.capacity() >= 100);
        v.push_back(0);
        assert(v.capacity() >= 101);
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, min_allocator<bool>> v;
        assert(v.capacity() == 0);
    }
    {
        std::vector<bool, min_allocator<bool>> v(100);
        assert(v.capacity() >= 100);
        v.push_back(0);
        assert(v.capacity() >= 101);
    }
#endif

  return 0;
}
