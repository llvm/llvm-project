//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// set& operator=(const set& s);

#include <set>
#include <cassert>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"

int main(int, char**)
{
    {
        typedef int V;
        V ar[] =
        {
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3
        };
        typedef test_less<int> C;
        typedef test_allocator<V> A;
        std::set<int, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(2));
        std::set<int, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A(7));
        m = mo;
        assert(m.get_allocator() == A(7));
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == 1);
        assert(*next(m.begin()) == 2);
        assert(*next(m.begin(), 2) == 3);

        assert(mo.get_allocator() == A(2));
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == 1);
        assert(*next(mo.begin()) == 2);
        assert(*next(mo.begin(), 2) == 3);
    }
    {
        typedef int V;
        const V ar[] =
        {
            1,
            2,
            3
        };
        std::set<int> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        std::set<int> *p = &m;
        m = *p;

        assert(m.size() == 3);
        assert(std::equal(m.begin(), m.end(), ar));
    }
    {
        typedef int V;
        V ar[] =
        {
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3
        };
        typedef test_less<int> C;
        typedef other_allocator<V> A;
        std::set<int, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(2));
        std::set<int, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0])/2, C(3), A(7));
        m = mo;
        assert(m.get_allocator() == A(2));
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == 1);
        assert(*next(m.begin()) == 2);
        assert(*next(m.begin(), 2) == 3);

        assert(mo.get_allocator() == A(2));
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 3);
        assert(distance(mo.begin(), mo.end()) == 3);
        assert(*mo.begin() == 1);
        assert(*next(mo.begin()) == 2);
        assert(*next(mo.begin(), 2) == 3);
    }

  return 0;
}
