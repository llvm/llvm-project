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

// void swap(basic_ostringstream& rhs);

#include <sstream>
#include <cassert>

int main(int, char**)
{
    {
        std::ostringstream ss0(" 123 456");
        std::ostringstream ss;
        ss.swap(ss0);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == " 123 456");
        int i = 234;
        ss << i << ' ' << 567;
        assert(ss.str() == "234 5676");
        ss0 << i << ' ' << 567;
        assert(ss0.str() == "234 567");
    }
    {
        std::wostringstream ss0(L" 123 456");
        std::wostringstream ss;
        ss.swap(ss0);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L" 123 456");
        int i = 234;
        ss << i << ' ' << 567;
        assert(ss.str() == L"234 5676");
        ss0 << i << ' ' << 567;
        assert(ss0.str() == L"234 567");
    }

  return 0;
}
