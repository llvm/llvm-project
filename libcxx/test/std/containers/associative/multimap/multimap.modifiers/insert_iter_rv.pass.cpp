//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// template <class P>
//     iterator insert(const_iterator position, P&& p);

#include <map>
#include <cassert>

#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef std::multimap<int, MoveOnly> M;
        typedef std::pair<int, MoveOnly> P;
        typedef M::iterator R;
        M m;
        R r = m.insert(m.cend(), P(2, 2));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(r->first == 2);
        assert(r->second == 2);

        r = m.insert(m.cend(), P(1, 1));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(r->first == 1);
        assert(r->second == 1);

        r = m.insert(m.cend(), P(3, 3));
        assert(r == prev(m.end()));
        assert(m.size() == 3);
        assert(r->first == 3);
        assert(r->second == 3);

        r = m.insert(m.cend(), P(3, 2));
        assert(r == prev(m.end()));
        assert(m.size() == 4);
        assert(r->first == 3);
        assert(r->second == 2);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::multimap<int, MoveOnly, std::less<int>, min_allocator<std::pair<const int, MoveOnly>>> M;
        typedef std::pair<int, MoveOnly> P;
        typedef M::iterator R;
        M m;
        R r = m.insert(m.cend(), P(2, 2));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(r->first == 2);
        assert(r->second == 2);

        r = m.insert(m.cend(), P(1, 1));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(r->first == 1);
        assert(r->second == 1);

        r = m.insert(m.cend(), P(3, 3));
        assert(r == prev(m.end()));
        assert(m.size() == 3);
        assert(r->first == 3);
        assert(r->second == 3);

        r = m.insert(m.cend(), P(3, 2));
        assert(r == prev(m.end()));
        assert(m.size() == 4);
        assert(r->first == 3);
        assert(r->second == 2);
    }
#endif
#if TEST_STD_VER > 14
    {
        typedef std::multimap<int, MoveOnly> M;
        typedef std::pair<int, MoveOnly> P;
        typedef M::iterator R;
        M m;
        R r = m.insert(m.cend(), {2, MoveOnly(2)});
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(r->first == 2);
        assert(r->second == 2);

        r = m.insert(m.cend(), {1, MoveOnly(1)});
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(r->first == 1);
        assert(r->second == 1);

        r = m.insert(m.cend(), {3, MoveOnly(3)});
        assert(r == prev(m.end()));
        assert(m.size() == 3);
        assert(r->first == 3);
        assert(r->second == 3);

        r = m.insert(m.cend(), {3, MoveOnly(2)});
        assert(r == prev(m.end()));
        assert(m.size() == 4);
        assert(r->first == 3);
        assert(r->second == 2);
    }
#endif
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
