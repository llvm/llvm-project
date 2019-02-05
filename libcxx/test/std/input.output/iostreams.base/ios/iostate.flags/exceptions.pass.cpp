//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// iostate exceptions() const;

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf : public std::streambuf {};

int main(int, char**)
{
    {
        const std::ios ios(0);
        assert(ios.exceptions() == std::ios::goodbit);
    }
    {
        testbuf sb;
        const std::ios ios(&sb);
        assert(ios.exceptions() == std::ios::goodbit);
    }

  return 0;
}
