//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void unique();

#include <list>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
    int a1[] = {2, 1, 1, 4, 4, 4, 4, 3, 3};
    int a2[] = {2, 1, 4, 3};
    std::list<int> c(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c.unique();
    assert(c == std::list<int>(a2, a2+4));
    }
#if TEST_STD_VER >= 11
    {
    int a1[] = {2, 1, 1, 4, 4, 4, 4, 3, 3};
    int a2[] = {2, 1, 4, 3};
    std::list<int, min_allocator<int>> c(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c.unique();
    assert((c == std::list<int, min_allocator<int>>(a2, a2+4)));
    }
#endif

  return 0;
}
