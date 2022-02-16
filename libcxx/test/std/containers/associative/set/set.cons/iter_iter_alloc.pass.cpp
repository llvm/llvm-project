//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// template <class InputIterator>
//     set(InputIterator first, InputIterator last,
//         const value_compare& comp, const allocator_type& a);
//
// template <class InputIterator>
//     set(InputIterator first, InputIterator last,
//         const allocator_type& a);

#include <set>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
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
    typedef test_less<V> C;
    typedef test_allocator<V> A;
    std::set<V, C, A> m(cpp17_input_iterator<const V*>(ar),
                        cpp17_input_iterator<const V*>(ar+sizeof(ar)/sizeof(ar[0])),
                        C(5), A(7));
    assert(m.value_comp() == C(5));
    assert(m.get_allocator() == A(7));
    assert(m.size() == 3);
    assert(std::distance(m.begin(), m.end()) == 3);
    assert(*m.begin() == 1);
    assert(*next(m.begin()) == 2);
    assert(*next(m.begin(), 2) == 3);
    }
#if TEST_STD_VER > 11
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
    typedef test_allocator<V> A;
    typedef test_less<int> C;
    A a(7);
    std::set<V, C, A> m(ar, ar+sizeof(ar)/sizeof(ar[0]), a);

    assert(m.size() == 3);
    assert(std::distance(m.begin(), m.end()) == 3);
    assert(*m.begin() == 1);
    assert(*next(m.begin()) == 2);
    assert(*next(m.begin(), 2) == 3);
    assert(m.get_allocator() == a);
    }
#endif

  return 0;
}
