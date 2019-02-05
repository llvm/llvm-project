//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void pop_back();

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"


int main(int, char**)
{
    {
        std::vector<int> c;
        c.push_back(1);
        assert(c.size() == 1);
        c.pop_back();
        assert(c.size() == 0);

    }
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> c;
        c.push_back(1);
        assert(c.size() == 1);
        c.pop_back();
        assert(c.size() == 0);
    }
#endif

  return 0;
}
