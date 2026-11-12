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

// int_type pbackfail(int_type c = traits::eof());

#include <streambuf>
#include <cassert>

#include "test_macros.h"

int pbackfail_called = 0;

struct test
    : public std::basic_streambuf<char>
{
    test() {}
};

int main(int, char**)
{
    test t;
    assert(t.sputbackc('A') == -1);

  return 0;
}
