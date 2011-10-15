//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

//       reference operator[](size_type __i);
// const_reference operator[](size_type __i) const;
//
//       reference at(size_type __i);
// const_reference at(size_type __i) const;
//
//       reference front();
// const_reference front() const;
//
//       reference back();
// const_reference back() const;

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

int main()
{
    {
        std::deque<int> c = make(10);
        for (unsigned i = 0; i < 10; ++i)
            assert(c[i] == i);
        for (unsigned i = 0; i < 10; ++i)
            assert(c.at(i) == i);
        assert(c.front() == 0);
        assert(c.back() == 9);
    }
    {
        const std::deque<int> c = make(10);
        for (unsigned i = 0; i < 10; ++i)
            assert(c[i] == i);
        for (unsigned i = 0; i < 10; ++i)
            assert(c.at(i) == i);
        assert(c.front() == 0);
        assert(c.back() == 9);
    }
}
