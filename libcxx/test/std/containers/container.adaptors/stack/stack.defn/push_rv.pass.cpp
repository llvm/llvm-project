//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <stack>

// void push(value_type&& v);

#include <stack>
#include <cassert>

#include "MoveOnly.h"

int main(int, char**)
{
    std::stack<MoveOnly> q;
    q.push(MoveOnly(1));
    assert(q.size() == 1);
    assert(q.top() == MoveOnly(1));
    q.push(MoveOnly(2));
    assert(q.size() == 2);
    assert(q.top() == MoveOnly(2));
    q.push(MoveOnly(3));
    assert(q.size() == 3);
    assert(q.top() == MoveOnly(3));

  return 0;
}
