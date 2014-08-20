//===--------------- catch_member_function_pointer_01.cpp -----------------===//
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
    void foo() {}
    void bar() const {}
};

typedef void (A::*mf1)();
typedef void (A::*mf2)() const;

void test1()
{
    try
    {
        throw &A::foo;
        assert(false);
    }
    catch (mf2)
    {
        assert(false);
    }
    catch (mf1)
    {
    }
}

void test2()
{
    try
    {
        throw &A::bar;
        assert(false);
    }
    catch (mf1)
    {
        assert(false);
    }
    catch (mf2)
    {
    }
}

int main()
{
    test1();
    test2();
}
