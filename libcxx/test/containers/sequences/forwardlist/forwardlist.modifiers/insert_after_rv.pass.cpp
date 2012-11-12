//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// iterator insert_after(const_iterator p, value_type&& v);

#include <forward_list>
#include <cassert>

#include "../../../MoveOnly.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef MoveOnly T;
        typedef std::forward_list<T> C;
        typedef C::iterator I;
        C c;
        I i = c.insert_after(c.cbefore_begin(), 0);
        assert(i == c.begin());
        assert(c.front() == 0);
        assert(c.front() == 0);
        assert(distance(c.begin(), c.end()) == 1);

        i = c.insert_after(c.cbegin(), 1);
        assert(i == next(c.begin()));
        assert(c.front() == 0);
        assert(*next(c.begin()) == 1);
        assert(distance(c.begin(), c.end()) == 2);

        i = c.insert_after(next(c.cbegin()), 2);
        assert(i == next(c.begin(), 2));
        assert(c.front() == 0);
        assert(*next(c.begin()) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(distance(c.begin(), c.end()) == 3);

        i = c.insert_after(c.cbegin(), 3);
        assert(i == next(c.begin()));
        assert(c.front() == 0);
        assert(*next(c.begin(), 1) == 3);
        assert(*next(c.begin(), 2) == 1);
        assert(*next(c.begin(), 3) == 2);
        assert(distance(c.begin(), c.end()) == 4);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
