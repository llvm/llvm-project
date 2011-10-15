//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class Compare> void sort(Compare comp);

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <vector>
#include <functional>
#include <cassert>

void test(int N)
{
    typedef int T;
    typedef std::forward_list<T> C;
    typedef std::vector<T> V;
    V v;
    for (int i = 0; i < N; ++i)
        v.push_back(i);
    std::random_shuffle(v.begin(), v.end());
    C c(v.begin(), v.end());
    c.sort(std::greater<T>());
    assert(distance(c.begin(), c.end()) == N);
    C::const_iterator j = c.begin();
    for (int i = 0; i < N; ++i, ++j)
        assert(*j == N-1-i);
}

int main()
{
    for (int i = 0; i < 40; ++i)
        test(i);
}
