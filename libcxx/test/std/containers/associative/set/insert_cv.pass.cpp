//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// pair<iterator, bool> insert(const value_type& v);

#include <set>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
        typedef std::set<int> M;
        typedef std::pair<M::iterator, bool> R;
        M m;
        R r = m.insert(M::value_type(2));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 1);
        assert(*r.first == 2);

        r = m.insert(M::value_type(1));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 2);
        assert(*r.first == 1);

        r = m.insert(M::value_type(3));
        assert(r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(*r.first == 3);

        r = m.insert(M::value_type(3));
        assert(!r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(*r.first == 3);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::set<int, std::less<int>, min_allocator<int>> M;
        typedef std::pair<M::iterator, bool> R;
        M m;
        R r = m.insert(M::value_type(2));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 1);
        assert(*r.first == 2);

        r = m.insert(M::value_type(1));
        assert(r.second);
        assert(r.first == m.begin());
        assert(m.size() == 2);
        assert(*r.first == 1);

        r = m.insert(M::value_type(3));
        assert(r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(*r.first == 3);

        r = m.insert(M::value_type(3));
        assert(!r.second);
        assert(r.first == prev(m.end()));
        assert(m.size() == 3);
        assert(*r.first == 3);
    }
#endif
}
