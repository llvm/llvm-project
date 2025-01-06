//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class map

// map(map&& m);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    typedef std::pair<const int, double> V;
    {
        typedef test_less<int> C;
        typedef test_allocator<V> A;
        std::map<int, double, C, A> mo(C(5), A(7));
        std::map<int, double, C, A> m = std::move(mo);
        assert(m.get_allocator() == A(7));
        assert(m.key_comp() == C(5));
        assert(m.size() == 0);
        assert(std::distance(m.begin(), m.end()) == 0);

        assert(mo.get_allocator() == A(7));
        assert(mo.get_allocator().get_id() == test_alloc_base::moved_value);
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 0);
        assert(std::distance(mo.begin(), mo.end()) == 0);
    }
    {
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2),
        };
        typedef test_less<int> C;
        typedef test_allocator<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A(7));
        std::map<int, double, C, A> m = std::move(mo);
        assert(m.get_allocator() == A(7));
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.get_allocator() == A(7));
        assert(mo.get_allocator().get_id() == test_alloc_base::moved_value);
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 0);
        assert(std::distance(mo.begin(), mo.end()) == 0);
    }
    {
        typedef test_less<int> C;
        typedef min_allocator<V> A;
        std::map<int, double, C, A> mo(C(5), A());
        std::map<int, double, C, A> m = std::move(mo);
        assert(m.get_allocator() == A());
        assert(m.key_comp() == C(5));
        assert(m.size() == 0);
        assert(std::distance(m.begin(), m.end()) == 0);

        assert(mo.get_allocator() == A());
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 0);
        assert(std::distance(mo.begin(), mo.end()) == 0);
    }
    {
        V ar[] =
        {
            V(1, 1),
            V(1, 1.5),
            V(1, 2),
            V(2, 1),
            V(2, 1.5),
            V(2, 2),
            V(3, 1),
            V(3, 1.5),
            V(3, 2),
        };
        typedef test_less<int> C;
        typedef min_allocator<V> A;
        std::map<int, double, C, A> mo(ar, ar+sizeof(ar)/sizeof(ar[0]), C(5), A());
        std::map<int, double, C, A> m = std::move(mo);
        assert(m.get_allocator() == A());
        assert(m.key_comp() == C(5));
        assert(m.size() == 3);
        assert(std::distance(m.begin(), m.end()) == 3);
        assert(*m.begin() == V(1, 1));
        assert(*std::next(m.begin()) == V(2, 1));
        assert(*std::next(m.begin(), 2) == V(3, 1));

        assert(mo.get_allocator() == A());
        assert(mo.key_comp() == C(5));
        assert(mo.size() == 0);
        assert(std::distance(mo.begin(), mo.end()) == 0);
    }

  return 0;
}
