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

// iterator insert(const_iterator position, const value_type& v);

#include <map>
#include <cassert>

int main()
{
    {
        typedef std::map<int, double> M;
        typedef M::iterator R;
        M m;
        R r = m.insert(m.end(), M::value_type(2, 2.5));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(r->first == 2);
        assert(r->second == 2.5);

        r = m.insert(m.end(), M::value_type(1, 1.5));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(r->first == 1);
        assert(r->second == 1.5);

        r = m.insert(m.end(), M::value_type(3, 3.5));
        assert(r == prev(m.end()));
        assert(m.size() == 3);
        assert(r->first == 3);
        assert(r->second == 3.5);

        r = m.insert(m.end(), M::value_type(3, 3.5));
        assert(r == prev(m.end()));
        assert(m.size() == 3);
        assert(r->first == 3);
        assert(r->second == 3.5);
    }
}
