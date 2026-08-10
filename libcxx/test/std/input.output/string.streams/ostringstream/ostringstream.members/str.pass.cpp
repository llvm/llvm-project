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

// void str(const basic_string<charT,traits,Allocator>& s);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ostringstream ss(" 123 456");
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == " 123 456");
        int i = 0;
        ss << i;
        assert(ss.str() == "0123 456");
        ss << 456;
        assert(ss.str() == "0456 456");
        ss.str(" 789");
        assert(ss.str() == " 789");
        ss << "abc";
        assert(ss.str() == "abc9");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wostringstream ss(L" 123 456");
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L" 123 456");
        int i = 0;
        ss << i;
        assert(ss.str() == L"0123 456");
        ss << 456;
        assert(ss.str() == L"0456 456");
        ss.str(L" 789");
        assert(ss.str() == L" 789");
        ss << L"abc";
        assert(ss.str() == L"abc9");
    }
#endif

  return 0;
}
