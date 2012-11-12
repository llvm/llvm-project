//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void reverse();

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <cassert>

void test(int N)
{
    typedef int T;
    typedef std::forward_list<T> C;
    C c;
    for (int i = 0; i < N; ++i)
        c.push_front(i);
    c.reverse();
    assert(distance(c.begin(), c.end()) == N);
    C::const_iterator j = c.begin();
    for (int i = 0; i < N; ++i, ++j)
        assert(*j == i);
}

int main()
{
    for (int i = 0; i < 10; ++i)
        test(i);
}
