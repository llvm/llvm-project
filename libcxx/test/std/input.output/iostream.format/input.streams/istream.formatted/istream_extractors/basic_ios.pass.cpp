//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_istream;

// basic_istream<charT,traits>& operator>>(basic_ios<charT,traits>&
//                                         (*pf)(basic_ios<charT,traits>&));

#include <istream>
#include <cassert>

int f_called = 0;

template <class CharT>
std::basic_ios<CharT>&
f(std::basic_ios<CharT>& is)
{
    ++f_called;
    return is;
}

int main(int, char**)
{
    {
        std::istream is((std::streambuf*)0);
        is >> f;
        assert(f_called == 1);
    }

  return 0;
}
