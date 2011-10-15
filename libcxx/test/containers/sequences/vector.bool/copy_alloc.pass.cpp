//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(const vector& v, const allocator_type& a);

#include <vector>
#include <cassert>
#include "../../test_allocator.h"

template <class C>
void
test(const C& x, const typename C::allocator_type& a)
{
    unsigned s = x.size();
    C c(x, a);
    assert(c.__invariants());
    assert(c.size() == s);
    assert(c == x);
}

int main()
{
    {
        int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
        int* an = a + sizeof(a)/sizeof(a[0]);
        test(std::vector<bool>(a, an), std::allocator<bool>());
    }
    {
        std::vector<bool, test_allocator<bool> > l(3, 2, test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > l2(l, test_allocator<bool>(3));
        assert(l2 == l);
        assert(l2.get_allocator() == test_allocator<bool>(3));
    }
    {
        std::vector<bool, other_allocator<bool> > l(3, 2, other_allocator<bool>(5));
        std::vector<bool, other_allocator<bool> > l2(l, other_allocator<bool>(3));
        assert(l2 == l);
        assert(l2.get_allocator() == other_allocator<bool>(3));
    }
}
