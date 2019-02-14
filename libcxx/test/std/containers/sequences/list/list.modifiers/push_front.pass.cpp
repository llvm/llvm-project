//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void push_front(const value_type& x);

#include <list>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
    std::list<int> c;
    for (int i = 0; i < 5; ++i)
        c.push_front(i);
    int a[] = {4, 3, 2, 1, 0};
    assert(c == std::list<int>(a, a+5));
    }
#if TEST_STD_VER >= 11
    {
    std::list<int, min_allocator<int>> c;
    for (int i = 0; i < 5; ++i)
        c.push_front(i);
    int a[] = {4, 3, 2, 1, 0};
    assert((c == std::list<int, min_allocator<int>>(a, a+5)));
    }
#endif

  return 0;
}
