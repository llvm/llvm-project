//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// priority_queue(const Compare& comp, const Container& c);

#include <queue>
#include <cassert>
#include <functional>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push_back(i);
    return c;
}

int main(int, char**)
{
    typedef std::vector<int> Container;
    typedef std::greater<int> Compare;
    typedef std::priority_queue<int, Container, Compare> Q;
    Container v = make<Container>(5);
    Q q(Compare(), v);
    assert(q.size() == 5);
    assert(q.top() == 0);

#if TEST_STD_VER >= 11
    // It should be explicit, so not convertible before C++20.
    static_assert(test_convertible<Q, const Compare&, const Container&>(), "");
#endif

    return 0;
}
