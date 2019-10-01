//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void pop_front();

#include <list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    int a[] = {1, 2, 3};
    std::list<int> c(a, a+3);
    c.pop_front();
    assert(c == std::list<int>(a+1, a+3));
    c.pop_front();
    assert(c == std::list<int>(a+2, a+3));
    c.pop_front();
    assert(c.empty());
    }
#if TEST_STD_VER >= 11
    {
    int a[] = {1, 2, 3};
    std::list<int, min_allocator<int>> c(a, a+3);
    c.pop_front();
    assert((c == std::list<int, min_allocator<int>>(a+1, a+3)));
    c.pop_front();
    assert((c == std::list<int, min_allocator<int>>(a+2, a+3)));
    c.pop_front();
    assert(c.empty());
    }
#endif

  return 0;
}
