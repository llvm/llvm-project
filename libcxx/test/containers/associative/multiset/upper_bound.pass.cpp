//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

//       iterator upper_bound(const key_type& k);
// const_iterator upper_bound(const key_type& k) const;

#include <set>
#include <cassert>

int main()
{
    typedef int V;
    typedef std::multiset<int> M;
    {
        typedef M::iterator R;
        V ar[] =
        {
            5,
            5,
            5,
            7,
            7,
            7,
            9,
            9,
            9
        };
        M m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        R r = m.upper_bound(4);
        assert(r == next(m.begin(), 0));
        r = m.upper_bound(5);
        assert(r == next(m.begin(), 3));
        r = m.upper_bound(6);
        assert(r == next(m.begin(), 3));
        r = m.upper_bound(7);
        assert(r == next(m.begin(), 6));
        r = m.upper_bound(8);
        assert(r == next(m.begin(), 6));
        r = m.upper_bound(9);
        assert(r == next(m.begin(), 9));
        r = m.upper_bound(11);
        assert(r == next(m.begin(), 9));
    }
    {
        typedef M::const_iterator R;
        V ar[] =
        {
            5,
            5,
            5,
            7,
            7,
            7,
            9,
            9,
            9
        };
        const M m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        R r = m.upper_bound(4);
        assert(r == next(m.begin(), 0));
        r = m.upper_bound(5);
        assert(r == next(m.begin(), 3));
        r = m.upper_bound(6);
        assert(r == next(m.begin(), 3));
        r = m.upper_bound(7);
        assert(r == next(m.begin(), 6));
        r = m.upper_bound(8);
        assert(r == next(m.begin(), 6));
        r = m.upper_bound(9);
        assert(r == next(m.begin(), 9));
        r = m.upper_bound(11);
        assert(r == next(m.begin(), 9));
    }
}
