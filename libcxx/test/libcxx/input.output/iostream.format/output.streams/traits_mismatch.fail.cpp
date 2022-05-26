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

// The char type of the stream and the char_type of the traits have to match

// UNSUPPORTED: no-wide-characters

#include <ostream>
#include <type_traits>
#include <cassert>

struct test_ostream
    : public std::basic_ostream<char, std::char_traits<wchar_t> > {};


int main(int, char**)
{
//  expected-error-re@ios:* {{static_assert failed{{.*}} "traits_type::char_type must be the same type as CharT"}}

  return 0;
}
