//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// char_type fill(char_type fillch);

#include <ios>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::ios ios(0);
    assert(ios.fill() == ' ');
    char c = ios.fill('*');
    assert(c == ' ');
    assert(ios.fill() == '*');

  return 0;
}
