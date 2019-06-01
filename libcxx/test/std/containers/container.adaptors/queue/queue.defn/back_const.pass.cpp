//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// const_reference back() const;

#include <queue>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::queue<int> q;
    assert(q.size() == 0);
    q.push(1);
    q.push(2);
    q.push(3);
    const std::queue<int>& cqr = q;
    const int& cir = cqr.back();
    assert(cir == 3);

  return 0;
}
