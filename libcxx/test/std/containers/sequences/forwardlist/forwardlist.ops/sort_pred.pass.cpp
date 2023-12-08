//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class Compare> void sort(Compare comp);

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <vector>
#include <functional>
#include <random>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

std::mt19937 randomness;

struct Payload
{
    int val;
    int side;
    Payload(int v) : val(v), side(0) {}
    Payload(int v, int s) : val(v), side(s) {}
    bool operator< (const Payload &rhs) const { return val <  rhs.val;}
};

bool greater(const Payload &lhs, const Payload &rhs) { return lhs.val >  rhs.val; }

void test_stable(int N)
{
    typedef Payload T;
    typedef std::forward_list<T> C;
    typedef std::vector<T> V;
    V v;
    for (int i = 0; i < N; ++i)
        v.push_back(Payload(i/2));
    std::shuffle(v.begin(), v.end(), randomness);
    for (int i = 0; i < N; ++i)
        v[i].side = i;

    C c(v.begin(), v.end());
    c.sort(greater);
    assert(std::distance(c.begin(), c.end()) == N);

//  Are we sorted?
    typename C::const_iterator j = c.begin();
    for (int i = 0; i < N; ++i, ++j)
        assert(j->val == (N-1-i)/2);

//  Are we stable?
    for (C::const_iterator it = c.begin(); it != c.end(); ++it)
    {
        C::const_iterator next = std::next(it);
        if (next != c.end() && it->val == next->val)
            assert(it->side < next->side);
    }
}

template <class C>
void test(int N)
{
    typedef typename C::value_type T;
    typedef std::vector<T> V;
    V v;
    for (int i = 0; i < N; ++i)
        v.push_back(i);
    std::shuffle(v.begin(), v.end(), randomness);
    C c(v.begin(), v.end());
    c.sort(std::greater<T>());
    assert(std::distance(c.begin(), c.end()) == N);
    typename C::const_iterator j = c.begin();
    for (int i = 0; i < N; ++i, ++j)
        assert(*j == N-1-i);
}

int main(int, char**)
{
    for (int i = 0; i < 40; ++i)
        test<std::forward_list<int> >(i);
#if TEST_STD_VER >= 11
    for (int i = 0; i < 40; ++i)
        test<std::forward_list<int, min_allocator<int>> >(i);
#endif

    for (int i = 0; i < 40; ++i)
        test_stable(i);

  return 0;
}
