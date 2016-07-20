//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// void push_back(value_type&& x);

#include <list>
#include <cassert>

#include "MoveOnly.h"
#include "min_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    std::list<MoveOnly> l1;
    l1.push_back(MoveOnly(1));
    assert(l1.size() == 1);
    assert(l1.front() == MoveOnly(1));
    l1.push_back(MoveOnly(2));
    assert(l1.size() == 2);
    assert(l1.front() == MoveOnly(1));
    assert(l1.back() == MoveOnly(2));
    }
#if TEST_STD_VER >= 11
    {
    std::list<MoveOnly, min_allocator<MoveOnly>> l1;
    l1.push_back(MoveOnly(1));
    assert(l1.size() == 1);
    assert(l1.front() == MoveOnly(1));
    l1.push_back(MoveOnly(2));
    assert(l1.size() == 2);
    assert(l1.front() == MoveOnly(1));
    assert(l1.back() == MoveOnly(2));
    }
#endif
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
