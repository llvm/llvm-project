//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector& operator=(vector&& c);

#include <vector>
#include <cassert>
#include "../../../MoveOnly.h"
#include "../../../test_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::vector<MoveOnly, test_allocator<MoveOnly> > l(test_allocator<MoveOnly>(5));
        std::vector<MoveOnly, test_allocator<MoveOnly> > lo(test_allocator<MoveOnly>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<MoveOnly, test_allocator<MoveOnly> > l2(test_allocator<MoveOnly>(5));
        l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
    {
        std::vector<MoveOnly, test_allocator<MoveOnly> > l(test_allocator<MoveOnly>(5));
        std::vector<MoveOnly, test_allocator<MoveOnly> > lo(test_allocator<MoveOnly>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<MoveOnly, test_allocator<MoveOnly> > l2(test_allocator<MoveOnly>(6));
        l2 = std::move(l);
        assert(l2 == lo);
        assert(!l.empty());
        assert(l2.get_allocator() == test_allocator<MoveOnly>(6));
    }
    {
        std::vector<MoveOnly, other_allocator<MoveOnly> > l(other_allocator<MoveOnly>(5));
        std::vector<MoveOnly, other_allocator<MoveOnly> > lo(other_allocator<MoveOnly>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<MoveOnly, other_allocator<MoveOnly> > l2(other_allocator<MoveOnly>(6));
        l2 = std::move(l);
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == lo.get_allocator());
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
