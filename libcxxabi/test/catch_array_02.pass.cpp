//===---------------------- catch_array_02.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can you have a catch clause of array type that catches anything?
// UNSUPPORTED: libcxxabi-no-exceptions

#include <cassert>

int main()
{
    typedef char Array[4];
    Array a = {'H', 'i', '!', 0};
    try
    {
        throw a;  // converts to char*
        assert(false);
    }
    catch (Array b)  // equivalent to char*
    {
    }
    catch (...)
    {
        assert(false);
    }
}
