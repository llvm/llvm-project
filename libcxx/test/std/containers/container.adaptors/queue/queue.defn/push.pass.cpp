//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// void push(const value_type& v);

#include <queue>
#include <cassert>

int main(int, char**)
{
    std::queue<int> q;
    q.push(1);
    assert(q.size() == 1);
    assert(q.front() == 1);
    assert(q.back() == 1);
    q.push(2);
    assert(q.size() == 2);
    assert(q.front() == 1);
    assert(q.back() == 2);
    q.push(3);
    assert(q.size() == 3);
    assert(q.front() == 1);
    assert(q.back() == 3);

  return 0;
}
