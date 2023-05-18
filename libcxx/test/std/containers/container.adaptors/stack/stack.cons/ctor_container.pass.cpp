//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// explicit stack(const Container&);

#include <stack>
#include <cassert>
#include <cstddef>

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
    typedef std::deque<int> Container;
    typedef std::stack<int> Q;
    Container d = make<Container>(5);
    Q q(d);
    assert(q.size() == 5);
    for (std::size_t i = 0; i < d.size(); ++i)
    {
        assert(q.top() == d[d.size() - i - 1]);
        q.pop();
    }

#if TEST_STD_VER >= 11
    static_assert(!test_convertible<Q, const Container&>(), "");
#endif

    return 0;
}
