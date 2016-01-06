//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// pair<iterator, bool> insert( value_type&& v);  // C++17 and later
// template <class P>
//   pair<iterator, bool> insert(P&& p);

#include <map>
#include <cassert>

#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef std::map<int, MoveOnly> M;
        typedef std::pair<M::iterator, bool> R;
        M m;
        R r = m.insert(M::value_type(2, 2));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 1);
        assert(r.first->first == 2);
        assert(r.first->second == 2);

        r = m.insert(M::value_type(1, 1));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 2);
        assert(r.first->first == 1);
        assert(r.first->second == 1);

        r = m.insert(M::value_type(3, 3));
        assert(r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(r.first->first == 3);
        assert(r.first->second == 3);

        r = m.insert(M::value_type(3, 3));
        assert(!r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(r.first->first == 3);
        assert(r.first->second == 3);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::map<int, MoveOnly, std::less<int>, min_allocator<std::pair<const int, MoveOnly>>> M;
        typedef std::pair<M::iterator, bool> R;
        M m;
        R r = m.insert(M::value_type(2, 2));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 1);
        assert(r.first->first == 2);
        assert(r.first->second == 2);

        r = m.insert(M::value_type(1, 1));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 2);
        assert(r.first->first == 1);
        assert(r.first->second == 1);

        r = m.insert(M::value_type(3, 3));
        assert(r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(r.first->first == 3);
        assert(r.first->second == 3);

        r = m.insert(M::value_type(3, 3));
        assert(!r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(r.first->first == 3);
        assert(r.first->second == 3);
    }
#endif
#if TEST_STD_VER > 14
    {
        typedef std::map<int, MoveOnly> M;
        typedef std::pair<M::iterator, bool> R;
        M m;
        R r = m.insert({2, MoveOnly(2)});
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 1);
        assert(r.first->first == 2);
        assert(r.first->second == 2);

        r = m.insert({1, MoveOnly(1)});
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 2);
        assert(r.first->first == 1);
        assert(r.first->second == 1);

        r = m.insert({3, MoveOnly(3)});
        assert(r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(r.first->first == 3);
        assert(r.first->second == 3);

        r = m.insert({3, MoveOnly(3)});
        assert(!r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(r.first->first == 3);
        assert(r.first->second == 3);
    }
#endif
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
