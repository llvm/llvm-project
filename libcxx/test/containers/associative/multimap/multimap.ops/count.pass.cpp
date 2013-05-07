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

// size_type count(const key_type& k) const;

#include <map>
#include <cassert>

int main()
{
    typedef std::pair<const int, double> V;
    typedef std::multimap<int, double> M;
    {
        typedef M::size_type R;
        V ar[] =
        {
            V(5, 1),
            V(5, 2),
            V(5, 3),
            V(7, 1),
            V(7, 2),
            V(7, 3),
            V(9, 1),
            V(9, 2),
            V(9, 3)
        };
        const M m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        R r = m.count(4);
        assert(r == 0);
        r = m.count(5);
        assert(r == 3);
        r = m.count(6);
        assert(r == 0);
        r = m.count(7);
        assert(r == 3);
        r = m.count(8);
        assert(r == 0);
        r = m.count(9);
        assert(r == 3);
        r = m.count(10);
        assert(r == 0);
    }
}
