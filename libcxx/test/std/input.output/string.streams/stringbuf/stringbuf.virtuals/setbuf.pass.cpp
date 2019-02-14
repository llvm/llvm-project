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

// basic_streambuf<charT,traits>* setbuf(charT* s, streamsize n);

#include <sstream>
#include <cassert>

int main(int, char**)
{
    {
        std::stringbuf sb("0123456789");
        assert(sb.pubsetbuf(0, 0) == &sb);
        assert(sb.str() == "0123456789");
    }
    {
        std::wstringbuf sb(L"0123456789");
        assert(sb.pubsetbuf(0, 0) == &sb);
        assert(sb.str() == L"0123456789");
    }

  return 0;
}
