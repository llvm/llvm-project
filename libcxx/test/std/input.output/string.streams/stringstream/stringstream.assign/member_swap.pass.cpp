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

// void swap(basic_stringstream& rhs);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::stringstream ss0(" 123 456 ");
        std::stringstream ss;
        ss.swap(ss0);
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
        ss0 << i << ' ' << 123;
        assert(ss0.str() == "456 123");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstringstream ss0(L" 123 456 ");
        std::wstringstream ss;
        ss.swap(ss0);
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
        ss0 << i << ' ' << 123;
        assert(ss0.str() == L"456 123");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
