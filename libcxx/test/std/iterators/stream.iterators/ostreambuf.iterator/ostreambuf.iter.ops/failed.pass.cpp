//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostreambuf_iterator

// bool failed() const throw();

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

template <typename Char, typename Traits = std::char_traits<Char> >
struct my_streambuf : public std::basic_streambuf<Char,Traits> {
    typedef typename std::basic_streambuf<Char,Traits>::int_type  int_type;
    typedef typename std::basic_streambuf<Char,Traits>::char_type char_type;

    my_streambuf() {}
    int_type sputc(char_type) { return Traits::eof(); }
    };

int main(int, char**)
{
    {
        my_streambuf<char> buf;
        std::ostreambuf_iterator<char> i(&buf);
        i = 'a';
        assert(i.failed());
    }
    {
        my_streambuf<wchar_t> buf;
        std::ostreambuf_iterator<wchar_t> i(&buf);
        i = L'a';
        assert(i.failed());
    }

  return 0;
}
