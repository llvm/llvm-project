//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// void push_back(const value_type& v);
// void pop_back();
// void pop_front();

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

void test(int size)
{
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2046, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int j = 0; j < N; ++j)
    {
        std::deque<int> c = make(size, rng[j]);
        std::deque<int>::const_iterator it = c.begin();
        for (int i = 0; i < size; ++i, ++it)
            assert(*it == i);
    }
}

int main()
{
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2046, 2047, 2048, 2049, 4094, 4095, 4096};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int j = 0; j < N; ++j)
        test(rng[j]);
}
