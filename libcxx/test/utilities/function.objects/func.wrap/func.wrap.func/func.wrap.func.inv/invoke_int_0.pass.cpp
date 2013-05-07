//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// R operator()(ArgTypes... args) const

#include <functional>
#include <cassert>

// 0 args, return int

int count = 0;

int f_int_0()
{
    return 3;
}

struct A_int_0
{
    int operator()() {return 4;}
};

void
test_int_0()
{
    // function
    {
    std::function<int ()> r1(f_int_0);
    assert(r1() == 3);
    }
    // function pointer
    {
    int (*fp)() = f_int_0;
    std::function<int ()> r1(fp);
    assert(r1() == 3);
    }
    // functor
    {
    A_int_0 a0;
    std::function<int ()> r1(a0);
    assert(r1() == 4);
    }
}

int main()
{
    test_int_0();
}
