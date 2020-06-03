//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// explicit priority_queue(const Compare& comp, container_type&& c);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"


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
    std::priority_queue<MoveOnly> q(std::less<MoveOnly>(), make<std::vector<MoveOnly> >(5));
    assert(q.size() == 5);
    assert(q.top() == MoveOnly(4));

  return 0;
}
