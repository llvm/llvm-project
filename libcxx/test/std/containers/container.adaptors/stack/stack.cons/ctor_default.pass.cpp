//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// stack();

#include <stack>
#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**)
{
    std::stack<int, std::vector<int, limited_allocator<int, 10> > > q;
    assert(q.size() == 0);
    q.push(1);
    q.push(2);
    assert(q.size() == 2);
    assert(q.top() == 2);

  return 0;
}
