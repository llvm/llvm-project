//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// iterator insert(const_iterator position, value_type&& x);

#if _LIBCPP_DEBUG >= 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))
#endif

#include <list>
#include <cassert>

#include "MoveOnly.h"
#include "min_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    std::list<MoveOnly> l1;
    l1.insert(l1.cend(), MoveOnly(1));
    assert(l1.size() == 1);
    assert(l1.front() == MoveOnly(1));
    l1.insert(l1.cbegin(), MoveOnly(2));
    assert(l1.size() == 2);
    assert(l1.front() == MoveOnly(2));
    assert(l1.back() == MoveOnly(1));
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
#if _LIBCPP_DEBUG >= 1
    {
        std::list<int> v1(3);
        std::list<int> v2(3);
        v1.insert(v2.begin(), 4);
        assert(false);
    }
#endif
#if TEST_STD_VER >= 11
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    std::list<MoveOnly, min_allocator<MoveOnly>> l1;
    l1.insert(l1.cend(), MoveOnly(1));
    assert(l1.size() == 1);
    assert(l1.front() == MoveOnly(1));
    l1.insert(l1.cbegin(), MoveOnly(2));
    assert(l1.size() == 2);
    assert(l1.front() == MoveOnly(2));
    assert(l1.back() == MoveOnly(1));
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
#if _LIBCPP_DEBUG >= 1
    {
        std::list<int, min_allocator<int>> v1(3);
        std::list<int, min_allocator<int>> v2(3);
        v1.insert(v2.begin(), 4);
        assert(false);
    }
#endif
#endif
}
