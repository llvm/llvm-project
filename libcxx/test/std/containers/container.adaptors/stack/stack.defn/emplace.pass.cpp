//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <stack>

// template <class... Args> reference emplace(Args&&... args);

#include <stack>
#include <cassert>

#include "../../../Emplaceable.h"

int main()
{
    typedef Emplaceable T;
    std::stack<Emplaceable> q;
    T& r1 = q.emplace(1, 2.5);
    assert(&r1 == &q.top());
    T& r2 = q.emplace(2, 3.5);
    assert(&r2 == &q.top());
    T& r3 = q.emplace(3, 4.5);
    assert(&r3 == &q.top());
    assert(q.size() == 3);
    assert(q.top() == Emplaceable(3, 4.5));
}
