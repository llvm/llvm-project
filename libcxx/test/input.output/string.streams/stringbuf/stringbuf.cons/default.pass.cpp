//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// explicit basic_stringbuf(ios_base::openmode which = ios_base::in | ios_base::out);

#include <sstream>
#include <cassert>

int main()
{
    {
        std::stringbuf buf;
        assert(buf.str() == "");
    }
    {
        std::wstringbuf buf;
        assert(buf.str() == L"");
    }
}
