//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// vector(const vector& v);

#include <vector>
#include <cassert>
#include "../../test_allocator.h"

template <class C>
void
test(const C& x)
{
    unsigned s = x.size();
    C c(x);
    assert(c.__invariants());
    assert(c.size() == s);
    assert(c == x);
}

int main()
{
    {
        bool a[] = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
        bool* an = a + sizeof(a)/sizeof(a[0]);
        test(std::vector<bool>(a, an));
    }
    {
        std::vector<bool, test_allocator<bool> > v(3, 2, test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > v2 = v;
        assert(v2 == v);
        assert(v2.get_allocator() == v.get_allocator());
    }
#ifndef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    {
        std::vector<bool, other_allocator<bool> > v(3, 2, other_allocator<bool>(5));
        std::vector<bool, other_allocator<bool> > v2 = v;
        assert(v2 == v);
        assert(v2.get_allocator() == other_allocator<bool>(-2));
    }
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE
}
