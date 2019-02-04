//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_ostringstream

// explicit basic_ostringstream(ios_base::openmode which = ios_base::in);

#include <sstream>
#include <cassert>

int main(int, char**)
{
    {
        std::ostringstream ss;
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
        std::ostringstream ss(std::ios_base::out);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
        std::wostringstream ss;
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
        std::wostringstream ss(std::ios_base::out);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L"");
    }

  return 0;
}
