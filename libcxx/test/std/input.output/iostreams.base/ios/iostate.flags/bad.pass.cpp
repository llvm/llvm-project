//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// bool bad() const;

#include <ios>
#include <streambuf>
#include <cassert>

#include "test_macros.h"

struct testbuf : public std::streambuf {};

int main(int, char**)
{
    {
        std::ios ios(0);
        assert(ios.bad());
        ios.setstate(std::ios::eofbit);
        assert(ios.bad());
    }
    {
        testbuf sb;
        std::ios ios(&sb);
        assert(!ios.bad());
        ios.setstate(std::ios::eofbit);
        assert(!ios.bad());
        ios.setstate(std::ios::failbit);
        assert(!ios.bad());
        ios.setstate(std::ios::badbit);
        assert(ios.bad());
    }

  return 0;
}
