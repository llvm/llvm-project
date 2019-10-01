//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringstream

// explicit basic_stringstream(ios_base::openmode which = ios_base::out|ios_base::in);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::stringstream ss;
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
        std::stringstream ss(std::ios_base::in);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
        std::wstringstream ss;
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
        std::wstringstream ss(std::ios_base::in);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L"");
    }

  return 0;
}
