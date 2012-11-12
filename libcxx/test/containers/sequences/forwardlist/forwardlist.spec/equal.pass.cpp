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
//     bool operator==(const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y);
//
// template <class T, class Allocator>
//     bool operator!=(const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y);

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <cassert>

void test(int N, int M)
{
    typedef int T;
    typedef std::forward_list<T> C;
    C c1;
    for (int i = 0; i < N; ++i)
        c1.push_front(i);
    C c2;
    for (int i = 0; i < M; ++i)
        c2.push_front(i);
    if (N == M)
        assert(c1 == c2);
    else
        assert(c1 != c2);
    c2 = c1;
    assert(c1 == c2);
    if (N > 0)
    {
        c2.front() = N+1;
        assert(c1 != c2);
    }
}

int main()
{
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            test(i, j);
}
