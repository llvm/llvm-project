//===----------------- catch_member_pointer_nullptr.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Catching an exception thrown as nullptr was not properly handled before
// 2f984cab4fa7, which landed in macOS 10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// UNSUPPORTED: no-exceptions

#include <cassert>

#if __has_feature(cxx_nullptr)

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
        throw nullptr;
        assert(false);
    }
    catch (md2 p)
    {
        assert(!p);
    }
    catch (md1)
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
    catch (md1 p)
    {
        assert(!p);
    }
    catch (md2)
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

int main(int, char**)
{
    test1();
    test2();

    return 0;
}
