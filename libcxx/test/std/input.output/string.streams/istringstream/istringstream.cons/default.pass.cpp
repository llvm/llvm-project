//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_istringstream

// explicit basic_istringstream(ios_base::openmode which = ios_base::in);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream ss;
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
        std::istringstream ss(std::ios_base::in);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
        std::wistringstream ss;
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
        std::wistringstream ss(std::ios_base::in);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L"");
    }

  return 0;
}
