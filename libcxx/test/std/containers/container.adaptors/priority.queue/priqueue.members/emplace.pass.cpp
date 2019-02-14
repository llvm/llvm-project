//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <queue>

// priority_queue();

// template <class... Args> void emplace(Args&&... args);

#include <queue>
#include <cassert>

#include "../../../Emplaceable.h"

int main(int, char**)
{
    std::priority_queue<Emplaceable> q;
    q.emplace(1, 2.5);
    assert(q.top() == Emplaceable(1, 2.5));
    q.emplace(3, 4.5);
    assert(q.top() == Emplaceable(3, 4.5));
    q.emplace(2, 3.5);
    assert(q.top() == Emplaceable(3, 4.5));

  return 0;
}
