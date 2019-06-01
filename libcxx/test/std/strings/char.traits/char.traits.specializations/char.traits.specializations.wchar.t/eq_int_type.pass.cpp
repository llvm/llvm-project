//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<wchar_t>

// static constexpr bool eq_int_type(int_type c1, int_type c2);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    assert( std::char_traits<wchar_t>::eq_int_type(L'a', L'a'));
    assert(!std::char_traits<wchar_t>::eq_int_type(L'a', L'A'));
    assert(!std::char_traits<wchar_t>::eq_int_type(std::char_traits<wchar_t>::eof(), L'A'));
    assert( std::char_traits<wchar_t>::eq_int_type(std::char_traits<wchar_t>::eof(),
                                                   std::char_traits<wchar_t>::eof()));

  return 0;
}
