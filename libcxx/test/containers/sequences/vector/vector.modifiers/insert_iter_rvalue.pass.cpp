//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// iterator insert(const_iterator position, value_type&& x);

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"
#include "../../../MoveOnly.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::vector<MoveOnly> v(100);
        std::vector<MoveOnly>::iterator i = v.insert(v.cbegin() + 10, MoveOnly(3));
        assert(v.size() == 101);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == MoveOnly());
        assert(v[j] == MoveOnly(3));
        for (++j; j < 101; ++j)
            assert(v[j] == MoveOnly());
    }
    {
        std::vector<MoveOnly, stack_allocator<MoveOnly, 300> > v(100);
        std::vector<MoveOnly, stack_allocator<MoveOnly, 300> >::iterator i = v.insert(v.cbegin() + 10, MoveOnly(3));
        assert(v.size() == 101);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == MoveOnly());
        assert(v[j] == MoveOnly(3));
        for (++j; j < 101; ++j)
            assert(v[j] == MoveOnly());
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
