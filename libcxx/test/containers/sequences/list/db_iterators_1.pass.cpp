//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// Compare iterators from different containers with == or !=.

#if _LIBCPP_DEBUG2 >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

int main()
{
    typedef int T;
    typedef std::list<T> C;
    C c1;
    C c2;
    bool b = c1.begin() != c2.begin();
    assert(false);
}

#else

int main()
{
}

#endif
