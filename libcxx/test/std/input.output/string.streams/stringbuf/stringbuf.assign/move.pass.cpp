//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// basic_stringbuf& operator=(basic_stringbuf&& rhs);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::stringbuf buf1("testing");
        std::stringbuf buf;
        buf = std::move(buf1);
        assert(buf.str() == "testing");
    }
    {
        std::stringbuf buf1("testing", std::ios_base::in);
        std::stringbuf buf;
        buf = std::move(buf1);
        assert(buf.str() == "testing");
    }
    {
        std::stringbuf buf1("testing", std::ios_base::out);
        std::stringbuf buf;
        buf = std::move(buf1);
        assert(buf.str() == "testing");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstringbuf buf1(L"testing");
        std::wstringbuf buf;
        buf = std::move(buf1);
        assert(buf.str() == L"testing");
    }
    {
        std::wstringbuf buf1(L"testing", std::ios_base::in);
        std::wstringbuf buf;
        buf = std::move(buf1);
        assert(buf.str() == L"testing");
    }
    {
        std::wstringbuf buf1(L"testing", std::ios_base::out);
        std::wstringbuf buf;
        buf = std::move(buf1);
        assert(buf.str() == L"testing");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
