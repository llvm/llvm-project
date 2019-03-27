//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<wchar_t>

// static constexpr bool lt(char_type c1, char_type c2);

#include <string>
#include <cassert>

int main(int, char**)
{
    assert(!std::char_traits<wchar_t>::lt(L'a', L'a'));
    assert( std::char_traits<wchar_t>::lt(L'A', L'a'));

  return 0;
}
