//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// void pop_back()

#include <deque>
#include <cassert>

std::deque<int>
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
    std::deque<int> c(init, 0);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(i);
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
};

void
test(std::deque<int>& c1)
{
    typedef std::deque<int> C;
    typedef C::iterator I;
    std::size_t c1_osize = c1.size();
    c1.pop_back();
    assert(c1.size() == c1_osize - 1);
    assert(distance(c1.begin(), c1.end()) == c1.size());
    I i = c1.begin();
    for (int j = 0; j < c1.size(); ++j, ++i)
        assert(*i == j);
}

void
testN(int start, int N)
{
    if (N != 0)
    {
        typedef std::deque<int> C;
        C c1 = make(N, start);
        test(c1);
    }
}

int main()
{
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            testN(rng[i], rng[j]);
}
