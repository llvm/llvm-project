//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// void*& pword(int idx);

#include <ios>
#include <string>
#include <cassert>

class test
    : public std::ios
{
public:
    test()
    {
        init(0);
    }
};

int main()
{
    test t;
    std::ios_base& b = t;
    for (int i = 0; i < 10000; ++i)
    {
        assert(b.pword(i) == 0);
        b.pword(i) = (void*)i;
        assert(b.pword(i) == (void*)i);
        for (int j = 0; j <= i; ++j)
            assert(b.pword(j) == (void*)j);
    }
}
