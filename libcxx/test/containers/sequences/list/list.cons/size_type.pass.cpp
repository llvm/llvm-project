//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// explicit list(size_type n);

#include <list>
#include <cassert>
#include "../../../DefaultOnly.h"
#include "../../../stack_allocator.h"

int main()
{
    {
        std::list<int> l(3);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 0);
        ++i;
        assert(*i == 0);
        ++i;
        assert(*i == 0);
    }
    {
        std::list<int, stack_allocator<int, 3> > l(3);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 0);
        ++i;
        assert(*i == 0);
        ++i;
        assert(*i == 0);
    }
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::list<DefaultOnly> l(3);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
