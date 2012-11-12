//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// explicit priority_queue(const Compare& comp, container_type&& c);

#include <queue>
#include <cassert>

#include "../../../MoveOnly.h"

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push_back(MoveOnly(i));
    return c;
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::priority_queue<MoveOnly> q(std::less<MoveOnly>(), make<std::vector<MoveOnly> >(5));
    assert(q.size() == 5);
    assert(q.top() == MoveOnly(4));
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
