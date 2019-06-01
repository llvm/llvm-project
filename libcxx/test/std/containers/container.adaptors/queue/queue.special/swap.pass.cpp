//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class T, class Container>
//   void swap(queue<T, Container>& x, queue<T, Container>& y);

#include <queue>
#include <cassert>

#include "test_macros.h"

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push(i);
    return c;
}

int main(int, char**)
{
    std::queue<int> q1 = make<std::queue<int> >(5);
    std::queue<int> q2 = make<std::queue<int> >(10);
    std::queue<int> q1_save = q1;
    std::queue<int> q2_save = q2;
    swap(q1, q2);
    assert(q1 == q2_save);
    assert(q2 == q1_save);

  return 0;
}
