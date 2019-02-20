//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ostream;

// explicit basic_ostream(basic_streambuf<charT,traits>* sb);

#include <ostream>
#include <cassert>

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    testbuf() {}
};

int main(int, char**)
{
    {
        testbuf<char> sb;
        std::basic_ostream<char> os(&sb);
        assert(os.rdbuf() == &sb);
        assert(os.tie() == 0);
        assert(os.fill() == ' ');
        assert(os.rdstate() == os.goodbit);
        assert(os.exceptions() == os.goodbit);
        assert(os.flags() == (os.skipws | os.dec));
        assert(os.precision() == 6);
        assert(os.getloc().name() == "C");
    }
    {
        testbuf<wchar_t> sb;
        std::basic_ostream<wchar_t> os(&sb);
        assert(os.rdbuf() == &sb);
        assert(os.tie() == 0);
        assert(os.fill() == L' ');
        assert(os.rdstate() == os.goodbit);
        assert(os.exceptions() == os.goodbit);
        assert(os.flags() == (os.skipws | os.dec));
        assert(os.precision() == 6);
        assert(os.getloc().name() == "C");
    }

  return 0;
}
