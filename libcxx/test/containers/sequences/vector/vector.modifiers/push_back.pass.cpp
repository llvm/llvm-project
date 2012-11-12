//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void push_back(const value_type& x);

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"

int main()
{
    {
        std::vector<int> c;
        c.push_back(0);
        assert(c.size() == 1);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(1);
        assert(c.size() == 2);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(2);
        assert(c.size() == 3);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(3);
        assert(c.size() == 4);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(4);
        assert(c.size() == 5);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
    }
    {
        std::vector<int, stack_allocator<int, 15> > c;
        c.push_back(0);
        assert(c.size() == 1);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(1);
        assert(c.size() == 2);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(2);
        assert(c.size() == 3);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(3);
        assert(c.size() == 4);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
        c.push_back(4);
        assert(c.size() == 5);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == j);
    }
}
