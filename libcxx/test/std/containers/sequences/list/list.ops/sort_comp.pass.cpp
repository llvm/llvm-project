//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <class Compare> sort(Compare comp);

#include <list>
#include <functional>
#include <algorithm>  // for is_permutation
#include <random>
#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

std::mt19937 randomness;

#ifndef TEST_HAS_NO_EXCEPTIONS
template <typename T>
struct throwingLess {
    throwingLess() : num_(1) {}
    throwingLess(int num) : num_(num) {}

    bool operator() (const T& lhs, const T& rhs) const
    {
    if ( --num_ == 0) throw 1;
    return lhs < rhs;
    }

    mutable int num_;
    };
#endif


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
    typedef std::list<T> C;
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


int main(int, char**)
{
    {
    int a1[] = {4, 8, 1, 0, 5, 7, 2, 3, 6, 11, 10, 9};
    int a2[] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::list<int> c1(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c1.sort(std::greater<int>());
    assert(c1 == std::list<int>(a2, a2+sizeof(a2)/sizeof(a2[0])));
    }

//  Test with throwing comparison; make sure that nothing is lost.
//  This is (sort of) LWG #2824
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
    int a1[] = {4, 8, 1, 0, 5, 7, 2, 3, 6, 11, 10, 9};
    const int sz = sizeof(a1)/sizeof(a1[0]);
    for (int i = 0; i < 10; ++i)
        {
        std::list<int> c1(a1, a1 + sz);
        try
            {
            throwingLess<int> comp(i);
            c1.sort(std::cref(comp));
            }
        catch (int) {}
        assert((c1.size() == sz));
        assert((std::is_permutation(c1.begin(), c1.end(), a1)));
        }
    }
#endif

#if TEST_STD_VER >= 11
    {
    int a1[] = {4, 8, 1, 0, 5, 7, 2, 3, 6, 11, 10, 9};
    int a2[] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    std::list<int, min_allocator<int>> c1(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c1.sort(std::greater<int>());
    assert((c1 == std::list<int, min_allocator<int>>(a2, a2+sizeof(a2)/sizeof(a2[0]))));
    }
#endif

    for (int i = 0; i < 40; ++i)
        test_stable(i);

  return 0;
}
