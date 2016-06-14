//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// Call erase(const_iterator position) with iterator from another container

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cassert>
#include <cstdlib>
#include <exception>

#include "min_allocator.h"

int main()
{
    {
    int a1[] = {1, 2, 3};
    std::list<int> l1(a1, a1+3);
    std::list<int> l2(a1, a1+3);
    std::list<int>::const_iterator i = l2.begin();
    l1.erase(i);
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    int a1[] = {1, 2, 3};
    std::list<int, min_allocator<int>> l1(a1, a1+3);
    std::list<int, min_allocator<int>> l2(a1, a1+3);
    std::list<int, min_allocator<int>>::const_iterator i = l2.begin();
    l1.erase(i);
    assert(false);
    }
#endif
}

#else

int main()
{
}

#endif
