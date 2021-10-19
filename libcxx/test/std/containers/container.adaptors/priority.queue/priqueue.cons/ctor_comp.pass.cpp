//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// explicit priority_queue(const Compare& comp);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

int main(int, char**)
{
    typedef std::vector<int, limited_allocator<int, 10> > Container;
    typedef std::less<int> Compare;
    typedef std::priority_queue<int, Container> Q;
    Q q((Compare()));
    assert(q.size() == 0);
    q.push(1);
    q.push(2);
    assert(q.size() == 2);
    assert(q.top() == 2);

#if TEST_STD_VER >= 11
    static_assert(!test_convertible<Q, const Compare&>(), "");
#endif

    return 0;
}
