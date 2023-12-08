//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// explicit queue(Container&& c = Container()); // before C++20
// queue() : queue(Container()) {}              // C++20
// explicit queue(Container&& c);               // C++20

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push_back(MoveOnly(i));
    return c;
}

int main(int, char**)
{
    typedef std::deque<MoveOnly> Container;
    typedef std::queue<MoveOnly> Q;
    Q q(make<std::deque<MoveOnly> >(5));
    assert(q.size() == 5);

#if TEST_STD_VER >= 11
    static_assert(!test_convertible<Q, Container&&>(), "");
#endif

    return 0;
}
