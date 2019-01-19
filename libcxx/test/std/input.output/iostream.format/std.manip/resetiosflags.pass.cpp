//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iomanip>

// T1 resetiosflags(ios_base::fmtflags mask);

#include <iomanip>
#include <istream>
#include <ostream>
#include <cassert>

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    testbuf() {}
};

int main()
{
    {
        testbuf<char> sb;
        std::istream is(&sb);
        assert(is.flags() & std::ios_base::skipws);
        is >> std::resetiosflags(std::ios_base::skipws);
        assert(!(is.flags() & std::ios_base::skipws));
    }
    {
        testbuf<char> sb;
        std::ostream os(&sb);
        assert(os.flags() & std::ios_base::skipws);
        os << std::resetiosflags(std::ios_base::skipws);
        assert(!(os.flags() & std::ios_base::skipws));
    }
    {
        testbuf<wchar_t> sb;
        std::wistream is(&sb);
        assert(is.flags() & std::ios_base::skipws);
        is >> std::resetiosflags(std::ios_base::skipws);
        assert(!(is.flags() & std::ios_base::skipws));
    }
    {
        testbuf<wchar_t> sb;
        std::wostream os(&sb);
        assert(os.flags() & std::ios_base::skipws);
        os << std::resetiosflags(std::ios_base::skipws);
        assert(!(os.flags() & std::ios_base::skipws));
    }
}
