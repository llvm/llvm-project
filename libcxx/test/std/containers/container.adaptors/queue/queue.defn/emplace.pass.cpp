//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <queue>

// template <class... Args> reference emplace(Args&&... args);
// return type is 'reference' in C++17; 'void' before


#include <queue>
#include <cassert>

#include "test_macros.h"

#include "../../../Emplaceable.h"

int main()
{
    typedef Emplaceable T;
    std::queue<Emplaceable> q;
#if TEST_STD_VER > 14
    T& r1 = q.emplace(1, 2.5);
    assert(&r1 == &q.back());
    T& r2 = q.emplace(2, 3.5);
    assert(&r2 == &q.back());
    T& r3 = q.emplace(3, 4.5);
    assert(&r3 == &q.back());
    assert(&r1 == &q.front());
#else
    q.emplace(1, 2.5);
    q.emplace(2, 3.5);
    q.emplace(3, 4.5);
#endif

    assert(q.size() == 3);
    assert(q.front() == Emplaceable(1, 2.5));
    assert(q.back() == Emplaceable(3, 4.5));
}
