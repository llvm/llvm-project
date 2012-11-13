//===------------------------- catch_ptr_02.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>

struct  A {};
A a;
const A ca = A();

void test1 ()
{
    try
    {
        throw &a;
        assert(false);
    }
    catch ( const A* )
    {
    }
    catch ( A *)
    {
        assert (false);
    }
}

void test2 ()
{
    try
     {
    	throw &a;
        assert(false);
    }
    catch ( A* )
    {
    }
    catch ( const A *)
    {
         assert (false);
    }
}

void test3 ()
{
    try
    {
        throw &ca;
        assert(false);
    }
    catch ( const A* )
    {
    }
    catch ( A *)
    {
        assert (false);
    }
}

void test4 ()
{
    try
    {
        throw &ca;
        assert(false);
    }
    catch ( A *)
    {
        assert (false);
    }
    catch ( const A* )
    {
    }
}

int main()
{
    test1();
    test2();
    test3();
    test4();
}
