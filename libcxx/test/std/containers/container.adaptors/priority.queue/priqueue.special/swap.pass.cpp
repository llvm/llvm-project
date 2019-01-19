//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// priority_queue();

// template <class T, class Container, class Compare>
//   void swap(priority_queue<T, Container, Compare>& x,
//             priority_queue<T, Container, Compare>& y);

#include <queue>
#include <cassert>

int main()
{
    std::priority_queue<int> q1;
    std::priority_queue<int> q2;
    q1.push(1);
    q1.push(3);
    q1.push(2);
    swap(q1, q2);
    assert(q1.empty());
    assert(q2.size() == 3);
    assert(q2.top() == 3);
}
