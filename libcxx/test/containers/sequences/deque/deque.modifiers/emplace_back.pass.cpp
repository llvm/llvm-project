//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// template <class... Args> void emplace_back(Args&&... args);

#include <deque>
#include <cassert>

#include "../../../Emplaceable.h"

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

std::deque<Emplaceable>
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
    std::deque<Emplaceable> c(init);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(Emplaceable());
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
};

void
test(std::deque<Emplaceable>& c1)
{
    typedef std::deque<Emplaceable> C;
    typedef C::iterator I;
    std::size_t c1_osize = c1.size();
    c1.emplace_back(Emplaceable(1, 2.5));
    assert(c1.size() == c1_osize + 1);
    assert(distance(c1.begin(), c1.end()) == c1.size());
    I i = c1.end();
    assert(*--i == Emplaceable(1, 2.5));
}

void
testN(int start, int N)
{
    typedef std::deque<Emplaceable> C;
    C c1 = make(N, start);
    test(c1);
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
