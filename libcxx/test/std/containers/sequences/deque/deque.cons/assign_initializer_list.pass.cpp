//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <deque>

// void assign(initializer_list<value_type> il);

#include <deque>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
    std::deque<int> d;
    d.assign({3, 4, 5, 6});
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
    }
    {
    std::deque<int, min_allocator<int>> d;
    d.assign({3, 4, 5, 6});
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
    }

  return 0;
}
