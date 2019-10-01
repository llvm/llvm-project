//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class T, class Allocator>
//     bool operator< (const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y);
//
// template <class T, class Allocator>
//     bool operator> (const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y);
//
// template <class T, class Allocator>
//     bool operator>=(const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y);
//
// template <class T, class Allocator>
//     bool operator<=(const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y);

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class C>
void test(int N, int M)
{
    C c1;
    for (int i = 0; i < N; ++i)
        c1.push_front(i);
    C c2;
    for (int i = 0; i < M; ++i)
        c2.push_front(i);
    if (N < M)
        assert(c1 < c2);
    if (N <= M)
        assert(c1 <= c2);
    if (N >= M)
        assert(c1 >= c2);
    if (N > M)
        assert(c1 > c2);
}

int main(int, char**)
{
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            test<std::forward_list<int> >(i, j);
#if TEST_STD_VER >= 11
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            test<std::forward_list<int, min_allocator<int>> >(i, j);
#endif

  return 0;
}
