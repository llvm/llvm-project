//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void push_back(value_type&& x);

#include <vector>
#include <cassert>
#include "../../../MoveOnly.h"
#include "../../../stack_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::vector<MoveOnly> c;
        c.push_back(MoveOnly(0));
        assert(c.size() == 1);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(1));
        assert(c.size() == 2);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(2));
        assert(c.size() == 3);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(3));
        assert(c.size() == 4);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(4));
        assert(c.size() == 5);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
    }
    {
        std::vector<MoveOnly, stack_allocator<MoveOnly, 15> > c;
        c.push_back(MoveOnly(0));
        assert(c.size() == 1);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(1));
        assert(c.size() == 2);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(2));
        assert(c.size() == 3);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(3));
        assert(c.size() == 4);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
        c.push_back(MoveOnly(4));
        assert(c.size() == 5);
        for (int j = 0; j < c.size(); ++j)
            assert(c[j] == MoveOnly(j));
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
