//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void pop_back();

#if _LIBCPP_DEBUG2 >= 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))
#endif

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"

#if _LIBCPP_DEBUG2 >= 1
#include <cstdlib>
#include <exception>

#endif

int main()
{
    {
        std::vector<int> c;
        c.push_back(1);
        assert(c.size() == 1);
        c.pop_back();
        assert(c.size() == 0);
#if _LIBCPP_DEBUG2 >= 1
        c.pop_back();
        assert(false);
#endif        
    }
}
