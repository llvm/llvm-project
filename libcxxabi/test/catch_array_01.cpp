//===---------------------- catch_array_01.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Can you have a catch clause of array type that catches anything?

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
    catch (Array& b)  // can't catch char*
    {
        assert(false);
    }
    catch (...)
    {
    }
}
