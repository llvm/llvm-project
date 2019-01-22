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

// basic_streambuf(const basic_streambuf& rhs);  // protected

#include <streambuf>
#include <cassert>

std::streambuf &get();

int main()
{
    std::streambuf sb = get(); // expected-error {{calling a protected constructor}}
}
