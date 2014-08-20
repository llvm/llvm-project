//===----------------- catch_member_data_pointer_01.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>

struct A
{
    const int i;
    int j;
};

typedef const int A::*md1;
typedef       int A::*md2;

void test1()
{
    try
    {
        throw &A::i;
        assert(false);
    }
    catch (md2)
    {
        assert(false);
    }
    catch (md1)
    {
    }
}

void test2()
{
    try
    {
        throw &A::j;
        assert(false);
    }
    catch (md1)
    {
        assert(false);
    }
    catch (md2)
    {
    }
}

int main()
{
    test1();
    test2();
}
