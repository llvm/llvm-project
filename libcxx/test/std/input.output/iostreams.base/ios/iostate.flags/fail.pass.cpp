//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// bool fail() const;

#include <ios>
#include <streambuf>
#include <cassert>

#include "test_macros.h"

struct testbuf : public std::streambuf {};

int main(int, char**)
{
    {
        std::ios ios(0);
        assert(ios.fail());
        ios.setstate(std::ios::eofbit);
        assert(ios.fail());
    }
    {
        testbuf sb;
        std::ios ios(&sb);
        assert(!ios.fail());
        ios.setstate(std::ios::eofbit);
        assert(!ios.fail());
        ios.setstate(std::ios::badbit);
        assert(ios.fail());
        ios.setstate(std::ios::failbit);
        assert(ios.fail());
    }

  return 0;
}
