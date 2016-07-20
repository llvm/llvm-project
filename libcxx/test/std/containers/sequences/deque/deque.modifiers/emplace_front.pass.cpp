//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// template <class... Args> void emplace_front(Args&&... args);

#include <deque>
#include <cassert>

#include "../../../Emplaceable.h"
#include "min_allocator.h"

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

template <class C>
C
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
    C c(init);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(Emplaceable());
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
}

template <class C>
void
test(C& c1)
{
    typedef typename C::iterator I;
    std::size_t c1_osize = c1.size();
    c1.emplace_front(Emplaceable(1, 2.5));
    assert(c1.size() == c1_osize + 1);
    assert(distance(c1.begin(), c1.end()) == c1.size());
    I i = c1.begin();
    assert(*i == Emplaceable(1, 2.5));
}

template <class C>
void
testN(int start, int N)
{
    C c1 = make<C>(N, start);
    test(c1);
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            testN<std::deque<Emplaceable> >(rng[i], rng[j]);
    }
#if TEST_STD_VER >= 11
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            testN<std::deque<Emplaceable, min_allocator<Emplaceable>> >(rng[i], rng[j]);
    }
#endif
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
