//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <stack>

// void push(value_type&& v);

#include <stack>
#include <cassert>

#include "../../../MoveOnly.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::stack<MoveOnly> q;
    q.push(MoveOnly(1));
    assert(q.size() == 1);
    assert(q.top() == MoveOnly(1));
    q.push(MoveOnly(2));
    assert(q.size() == 2);
    assert(q.top() == MoveOnly(2));
    q.push(MoveOnly(3));
    assert(q.size() == 3);
    assert(q.top() == MoveOnly(3));
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
