//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// bool eof() const;

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf : public std::streambuf {};

int main()
{
    {
        std::ios ios(0);
        assert(!ios.eof());
        ios.setstate(std::ios::eofbit);
        assert(ios.eof());
    }
    {
        testbuf sb;
        std::ios ios(&sb);
        assert(!ios.eof());
        ios.setstate(std::ios::eofbit);
        assert(ios.eof());
    }
}
