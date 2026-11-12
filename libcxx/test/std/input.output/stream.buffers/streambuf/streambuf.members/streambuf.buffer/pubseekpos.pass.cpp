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

// pos_type pubseekpos(pos_type sp,
//                     ios_base::openmode which = ios_base::in | ios_base::out;

#include <streambuf>
#include <cassert>

#include "test_macros.h"

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    test() {}
};

int main(int, char**)
{
    {
        test<char> t;
        assert(t.pubseekpos(0, std::ios_base::app) == -1);
    }

  return 0;
}
