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

// explicit basic_stringbuf(const basic_string<charT,traits,Allocator>& s,
//                          ios_base::openmode which = ios_base::in | ios_base::out);

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::stringbuf buf("testing");
        assert(buf.str() == "testing");
    }
    {
        std::stringbuf buf("testing", std::ios_base::in);
        assert(buf.str() == "testing");
    }
    {
        std::stringbuf buf("testing", std::ios_base::out);
        assert(buf.str() == "testing");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstringbuf buf(L"testing");
        assert(buf.str() == L"testing");
    }
    {
        std::wstringbuf buf(L"testing", std::ios_base::in);
        assert(buf.str() == L"testing");
    }
    {
        std::wstringbuf buf(L"testing", std::ios_base::out);
        assert(buf.str() == L"testing");
    }
#endif

  return 0;
}
