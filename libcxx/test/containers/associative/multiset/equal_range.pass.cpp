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

// pair<iterator,iterator>             equal_range(const key_type& k);
// pair<const_iterator,const_iterator> equal_range(const key_type& k) const;

#include <set>
#include <cassert>

int main()
{
    typedef int V;
    typedef std::multiset<int> M;
    {
        typedef std::pair<M::iterator, M::iterator> R;
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
        R r = m.equal_range(4);
        assert(r.first  == next(m.begin(), 0));
        assert(r.second == next(m.begin(), 0));
        r = m.equal_range(5);
        assert(r.first  == next(m.begin(), 0));
        assert(r.second == next(m.begin(), 3));
        r = m.equal_range(6);
        assert(r.first  == next(m.begin(), 3));
        assert(r.second == next(m.begin(), 3));
        r = m.equal_range(7);
        assert(r.first  == next(m.begin(), 3));
        assert(r.second == next(m.begin(), 6));
        r = m.equal_range(8);
        assert(r.first  == next(m.begin(), 6));
        assert(r.second == next(m.begin(), 6));
        r = m.equal_range(9);
        assert(r.first  == next(m.begin(), 6));
        assert(r.second == next(m.begin(), 9));
        r = m.equal_range(10);
        assert(r.first  == next(m.begin(), 9));
        assert(r.second == next(m.begin(), 9));
    }
    {
        typedef std::pair<M::const_iterator, M::const_iterator> R;
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
        R r = m.equal_range(4);
        assert(r.first  == next(m.begin(), 0));
        assert(r.second == next(m.begin(), 0));
        r = m.equal_range(5);
        assert(r.first  == next(m.begin(), 0));
        assert(r.second == next(m.begin(), 3));
        r = m.equal_range(6);
        assert(r.first  == next(m.begin(), 3));
        assert(r.second == next(m.begin(), 3));
        r = m.equal_range(7);
        assert(r.first  == next(m.begin(), 3));
        assert(r.second == next(m.begin(), 6));
        r = m.equal_range(8);
        assert(r.first  == next(m.begin(), 6));
        assert(r.second == next(m.begin(), 6));
        r = m.equal_range(9);
        assert(r.first  == next(m.begin(), 6));
        assert(r.second == next(m.begin(), 9));
        r = m.equal_range(10);
        assert(r.first  == next(m.begin(), 9));
        assert(r.second == next(m.begin(), 9));
    }
}
