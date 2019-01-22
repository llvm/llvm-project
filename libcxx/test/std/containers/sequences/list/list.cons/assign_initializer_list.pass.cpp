//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <list>

// void assign(initializer_list<value_type> il);

#include <list>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
    std::list<int> d;
    d.assign({3, 4, 5, 6});
    assert(d.size() == 4);
    std::list<int>::iterator i = d.begin();
    assert(*i++ == 3);
    assert(*i++ == 4);
    assert(*i++ == 5);
    assert(*i++ == 6);
    }
    {
    std::list<int, min_allocator<int>> d;
    d.assign({3, 4, 5, 6});
    assert(d.size() == 4);
    std::list<int, min_allocator<int>>::iterator i = d.begin();
    assert(*i++ == 3);
    assert(*i++ == 4);
    assert(*i++ == 5);
    assert(*i++ == 6);
    }
}
