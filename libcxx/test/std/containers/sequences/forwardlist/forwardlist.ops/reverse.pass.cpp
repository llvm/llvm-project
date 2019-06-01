//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void reverse();

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class C>
void test(int N)
{
    C c;
    for (int i = 0; i < N; ++i)
        c.push_front(i);
    c.reverse();
    assert(distance(c.begin(), c.end()) == N);
    typename C::const_iterator j = c.begin();
    for (int i = 0; i < N; ++i, ++j)
        assert(*j == i);
}

int main(int, char**)
{
    for (int i = 0; i < 10; ++i)
        test<std::forward_list<int> >(i);
#if TEST_STD_VER >= 11
    for (int i = 0; i < 10; ++i)
        test<std::forward_list<int, min_allocator<int>> >(i);
#endif

  return 0;
}
