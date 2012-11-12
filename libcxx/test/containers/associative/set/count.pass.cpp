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

// size_type count(const key_type& k) const;

#include <set>
#include <cassert>

int main()
{
    typedef int V;
    typedef std::set<int> M;
    {
        typedef M::size_type R;
        V ar[] =
        {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
        };
        const M m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        R r = m.count(5);
        assert(r == 1);
        r = m.count(6);
        assert(r == 1);
        r = m.count(7);
        assert(r == 1);
        r = m.count(8);
        assert(r == 1);
        r = m.count(9);
        assert(r == 1);
        r = m.count(10);
        assert(r == 1);
        r = m.count(11);
        assert(r == 1);
        r = m.count(12);
        assert(r == 1);
        r = m.count(4);
        assert(r == 0);
    }
}
