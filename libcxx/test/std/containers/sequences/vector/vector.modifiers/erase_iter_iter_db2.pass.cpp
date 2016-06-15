//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// Call erase(const_iterator first, const_iterator last); with second iterator from another container

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <vector>
#include <cassert>
#include <exception>
#include <cstdlib>

#include "min_allocator.h"

int main()
{
    {
    int a1[] = {1, 2, 3};
    std::vector<int> l1(a1, a1+3);
    std::vector<int> l2(a1, a1+3);
    std::vector<int>::iterator i = l1.erase(l1.cbegin(), l2.cbegin()+1);
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    int a1[] = {1, 2, 3};
    std::vector<int, min_allocator<int>> l1(a1, a1+3);
    std::vector<int, min_allocator<int>> l2(a1, a1+3);
    std::vector<int, min_allocator<int>>::iterator i = l1.erase(l1.cbegin(), l2.cbegin()+1);
    assert(false);
    }
#endif
}

#else

int main()
{
}

#endif
