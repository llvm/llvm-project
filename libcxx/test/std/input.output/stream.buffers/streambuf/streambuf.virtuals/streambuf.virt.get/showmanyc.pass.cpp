//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// streamsize showmanyc();

#include <streambuf>
#include <cassert>

int showmanyc_called = 0;

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    test() {}
};

int main()
{
    test<char> t;
    assert(t.in_avail() == 0);
}
