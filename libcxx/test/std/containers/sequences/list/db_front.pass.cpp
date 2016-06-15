//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// Call front() on empty container.

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

#include "min_allocator.h"

int main()
{
    {
    typedef int T;
    typedef std::list<T> C;
    C c(1);
    assert(c.front() == 0);
    c.clear();
    assert(c.front() == 0);
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    typedef int T;
    typedef std::list<T, min_allocator<T>> C;
    C c(1);
    assert(c.front() == 0);
    c.clear();
    assert(c.front() == 0);
    assert(false);
    }
#endif
}

#else

int main()
{
}

#endif
