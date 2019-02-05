//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <queue>

// queue(queue&& q);

#include <queue>
#include <cassert>

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
    std::queue<MoveOnly> q(make<std::deque<MoveOnly> >(5));
    std::queue<MoveOnly> q2 = std::move(q);
    assert(q2.size() == 5);
    assert(q.empty());

  return 0;
}
