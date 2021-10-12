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

// template <class charT, class traits, class Allocator>
//   void
//   swap(basic_istringstream<charT, traits, Allocator>& x,
//        basic_istringstream<charT, traits, Allocator>& y);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream ss0(" 123 456");
        std::istringstream ss(" 789 321");
        swap(ss, ss0);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == " 123 456");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
        ss0 >> i;
        assert(i == 789);
        ss0 >> i;
        assert(i == 321);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wistringstream ss0(L" 123 456");
        std::wistringstream ss(L" 789 321");
        swap(ss, ss0);
        assert(ss.rdbuf() != 0);
        assert(ss.good());
        assert(ss.str() == L" 123 456");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
        ss0 >> i;
        assert(i == 789);
        ss0 >> i;
        assert(i == 321);
    }
#endif

  return 0;
}
