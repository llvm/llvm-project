//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// list(size_type n, const T& value, const Allocator& = Allocator());

#include <list>
#include <cassert>
#include "DefaultOnly.h"
#include "../../../stack_allocator.h"
#include "min_allocator.h"

int main()
{
    {
        std::list<int> l(3, 2);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
    {
        std::list<int> l(3, 2, std::allocator<int>());
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
    {
        std::list<int, stack_allocator<int, 3> > l(3, 2);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
#if TEST_STD_VER >= 11
    {
        std::list<int, min_allocator<int>> l(3, 2);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int, min_allocator<int>>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
    {
        std::list<int, min_allocator<int>> l(3, 2, min_allocator<int>());
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int, min_allocator<int>>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
#endif
}
