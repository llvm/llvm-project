//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// static constexpr bool eq(char_type c1, char_type c2);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    assert(std::char_traits<char>::eq('a', 'a'));
    assert(!std::char_traits<char>::eq('a', 'A'));

  return 0;
}
