//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void      unique(); // before C++20
// size_type unique(); // C++20 and later

#include <list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    int a1[] = {2, 1, 1, 4, 4, 4, 4, 3, 3};
    int a2[] = {2, 1, 4, 3};
    typedef std::list<int> L;
    L c(a1, a1+sizeof(a1)/sizeof(a1[0]));
#if TEST_STD_VER > 17
	ASSERT_SAME_TYPE(L::size_type, decltype(c.unique()));
    assert(c.unique() == 5);
#else
	ASSERT_SAME_TYPE(void,         decltype(c.unique()));
    c.unique();
#endif
    assert(c == std::list<int>(a2, a2+4));
    }
#if TEST_STD_VER >= 11
    {
    int a1[] = {2, 1, 1, 4, 4, 4, 4, 3, 3};
    int a2[] = {2, 1, 4, 3};
    std::list<int, min_allocator<int>> c(a1, a1+sizeof(a1)/sizeof(a1[0]));
#if TEST_STD_VER > 17
    assert(c.unique() == 5);
#else
    c.unique();
#endif
    assert((c == std::list<int, min_allocator<int>>(a2, a2+4)));
    }
#endif

  return 0;
}
