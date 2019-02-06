//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// const_reference top() const;

#include <stack>
#include <cassert>

int main()
{
    std::stack<int> q;
    assert(q.size() == 0);
    q.push(1);
    q.push(2);
    q.push(3);
    const std::stack<int>& cqr = q;
    const int& cir = cqr.top();
    assert(cir == 3);
}
