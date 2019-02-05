//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_istream;

// basic_istream<charT,traits>& operator>>(ios_base& (*pf)(ios_base&));

#include <istream>
#include <cassert>

int f_called = 0;

std::ios_base&
f(std::ios_base& is)
{
    ++f_called;
    return is;
}

int main()
{
    {
        std::istream is((std::streambuf*)0);
        is >> f;
        assert(f_called == 1);
    }
}
