//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// Clang and GCC emit warnings about exceptions of type 'Child' being caught by
// an earlier handler of type 'Base'.
// ADDITIONAL_COMPILE_FLAGS: -Wno-exceptions

#include <cassert>

struct A {};

void test1()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A* p)
    {
        assert(!p);
    }
    catch (const A*)
    {
        assert(false);
    }
}


void test2()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (const A* p)
    {
        assert(!p);
    }
    catch (A*)
    {
        assert(false);
    }
}

void test3()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (const A* const p)
    {
        assert(!p);
    }
    catch (A*)
    {
        assert(false);
    }
}

void test4()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A* p)
    {
        assert(!p);
    }
    catch (const A* const)
    {
        assert(false);
    }
}

void test5()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A const* p)
    {
        assert(!p);
    }
    catch (A*)
    {
        assert(false);
    }
}

void test6()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A* p)
    {
        assert(!p);
    }
    catch (A const*)
    {
        assert(false);
    }
}

int main(int, char**) {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();

    return 0;
}
