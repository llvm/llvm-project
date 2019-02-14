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

// explicit basic_istringstream(const basic_string<charT,traits,allocator>& str,
//                              ios_base::openmode which = ios_base::in);

#include <sstream>
#include <cassert>

int main(int, char**)
{
    {
        std::istringstream ss(" 123 456");
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == " 123 456");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
    }
    {
        std::istringstream ss(" 123 456", std::ios_base::out);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == " 123 456");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
    }
    {
        std::wistringstream ss(L" 123 456");
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L" 123 456");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
    }
    {
        std::wistringstream ss(L" 123 456", std::ios_base::out);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L" 123 456");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
    }

  return 0;
}
