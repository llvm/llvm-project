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
// return type is 'reference' in C++17; 'void' before

#include <stack>
#include <cassert>

#include "test_macros.h"

#include "../../../Emplaceable.h"

int main()
{
    typedef Emplaceable T;
    std::stack<Emplaceable> q;
#if TEST_STD_VER > 14
    T& r1 = q.emplace(1, 2.5);
    assert(&r1 == &q.top());
    T& r2 = q.emplace(2, 3.5);
    assert(&r2 == &q.top());
    T& r3 = q.emplace(3, 4.5);
    assert(&r3 == &q.top());
#else
    q.emplace(1, 2.5);
    q.emplace(2, 3.5);
    q.emplace(3, 4.5);
#endif
    assert(q.size() == 3);
    assert(q.top() == Emplaceable(3, 4.5));
}
