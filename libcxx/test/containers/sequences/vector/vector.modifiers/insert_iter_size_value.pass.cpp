//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// iterator insert(const_iterator position, size_type n, const value_type& x);

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"

int main()
{
    {
        std::vector<int> v(100);
        std::vector<int>::iterator i = v.insert(v.cbegin() + 10, 5, 1);
        assert(v.size() == 105);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (; j < 15; ++j)
            assert(v[j] == 1);
        for (++j; j < 105; ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<int, stack_allocator<int, 300> > v(100);
        std::vector<int, stack_allocator<int, 300> >::iterator i = v.insert(v.cbegin() + 10, 5, 1);
        assert(v.size() == 105);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (; j < 15; ++j)
            assert(v[j] == 1);
        for (++j; j < 105; ++j)
            assert(v[j] == 0);
    }
}
