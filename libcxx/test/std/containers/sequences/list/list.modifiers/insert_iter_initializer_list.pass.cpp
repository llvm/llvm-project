//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <list>

// iterator insert(const_iterator p, initializer_list<value_type> il);

#include <list>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
    std::list<int> d(10, 1);
    std::list<int>::iterator i = d.insert(next(d.cbegin(), 2), {3, 4, 5, 6});
    assert(d.size() == 14);
    assert(i == next(d.begin(), 2));
    i = d.begin();
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 3);
    assert(*i++ == 4);
    assert(*i++ == 5);
    assert(*i++ == 6);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    }
    {
    std::list<int, min_allocator<int>> d(10, 1);
    std::list<int, min_allocator<int>>::iterator i = d.insert(next(d.cbegin(), 2), {3, 4, 5, 6});
    assert(d.size() == 14);
    assert(i == next(d.begin(), 2));
    i = d.begin();
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 3);
    assert(*i++ == 4);
    assert(*i++ == 5);
    assert(*i++ == 6);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    }

  return 0;
}
