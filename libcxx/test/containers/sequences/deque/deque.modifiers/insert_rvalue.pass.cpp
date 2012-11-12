//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// iterator insert (const_iterator p, value_type&& v);

#include <deque>
#include <cassert>

#include "../../../MoveOnly.h"

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

std::deque<MoveOnly>
make(int size, int start = 0 )
{
    const int b = 4096 / sizeof(int);
    int init = 0;
    if (start > 0)
    {
        init = (start+1) / b + ((start+1) % b != 0);
        init *= b;
        --init;
    }
    std::deque<MoveOnly> c(init);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(MoveOnly(i));
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
};

void
test(int P, std::deque<MoveOnly>& c1, int x)
{
    typedef std::deque<MoveOnly> C;
    typedef C::iterator I;
    typedef C::const_iterator CI;
    std::size_t c1_osize = c1.size();
    CI i = c1.insert(c1.begin() + P, MoveOnly(x));
    assert(i == c1.begin() + P);
    assert(c1.size() == c1_osize + 1);
    assert(distance(c1.begin(), c1.end()) == c1.size());
    i = c1.begin();
    for (int j = 0; j < P; ++j, ++i)
        assert(*i == MoveOnly(j));
    assert(*i == MoveOnly(x));
    ++i;
    for (int j = P; j < c1_osize; ++j, ++i)
        assert(*i == MoveOnly(j));
}

void
testN(int start, int N)
{
    typedef std::deque<MoveOnly> C;
    typedef C::iterator I;
    typedef C::const_iterator CI;
    for (int i = 0; i <= 3; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make(N, start);
            test(i, c1, -10);
        }
    }
    for (int i = N/2-1; i <= N/2+1; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make(N, start);
            test(i, c1, -10);
        }
    }
    for (int i = N - 3; i <= N; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make(N, start);
            test(i, c1, -10);
        }
    }
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            testN(rng[i], rng[j]);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
