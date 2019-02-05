//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void merge(forward_list&& x);

#include <forward_list>
#include <iterator>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {3, 5, 6, 7, 12, 13};
        const T t2[] = {0, 1, 2, 4, 8, 9, 10, 11, 14, 15};
        const T t3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        c1.merge(c2);
        C c3(std::begin(t3), std::end(t3));
        assert(c1 == c3);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {3, 5, 6, 7, 12, 13};
        const T t2[] = {0, 1, 2, 4, 8, 9, 10, 11, 14, 15};
        const T t3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        c1.merge(c2);
        C c3(std::begin(t3), std::end(t3));
        assert(c1 == c3);
    }
#endif

  return 0;
}
