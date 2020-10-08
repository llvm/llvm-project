//===---------------------- catch_array_01.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can you have a catch clause of array type that catches anything?

// GCC incorrectly allows array types to be caught by reference.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69372
// XFAIL: gcc
// UNSUPPORTED: no-exceptions

#include <cassert>

int main(int, char**)
{
    typedef char Array[4];
    Array a = {'H', 'i', '!', 0};
    try
    {
        throw a;  // converts to char*
        assert(false);
    }
    catch (Array& b)  // can't catch char*
    {
        assert(false);
    }
    catch (...)
    {
    }

    return 0;
}
