//===--------------------- catch_pointer_nullptr.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>

#if __has_feature(cxx_nullptr)

void test1()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (int*)
    {
    }
    catch (long*)
    {
        assert(false);
    }
}

struct A {};

void test2()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A*)
    {
    }
    catch (int*)
    {
        assert(false);
    }
}

#else

void test1()
{
}

void test2()
{
}

#endif

int main()
{
    test1();
    test2();
}
