//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// basic_streambuf<charT,traits>* rdbuf() const;

#include <ios>
#include <streambuf>
#include <cassert>

int main()
{
    {
        const std::ios ios(0);
        assert(ios.rdbuf() == 0);
    }
    {
        std::streambuf* sb = (std::streambuf*)1;
        const std::ios ios(sb);
        assert(ios.rdbuf() == sb);
    }
}
