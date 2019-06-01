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

// void str(const basic_string<charT,traits,Allocator>& str);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::stringstream ss(" 123 456 ");
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == " 123 456 ");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
        ss << i << ' ' << 123;
        assert(ss.str() == "456 1236 ");
        ss.str("5466 89 ");
        ss >> i;
        assert(i == 5466);
        ss >> i;
        assert(i == 89);
        ss << i << ' ' << 321;
        assert(ss.str() == "89 3219 ");
    }
    {
        std::wstringstream ss(L" 123 456 ");
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L" 123 456 ");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
        ss << i << ' ' << 123;
        assert(ss.str() == L"456 1236 ");
        ss.str(L"5466 89 ");
        ss >> i;
        assert(i == 5466);
        ss >> i;
        assert(i == 89);
        ss << i << ' ' << 321;
        assert(ss.str() == L"89 3219 ");
    }
    {
        std::stringstream ss;
        ss.write("\xd1", 1);
        assert(ss.str().length() == 1);
    }

  return 0;
}
