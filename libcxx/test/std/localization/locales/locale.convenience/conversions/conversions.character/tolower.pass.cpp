//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> charT tolower(charT c, const locale& loc);

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l;
    assert(std::tolower(' ', l) == ' ');
    assert(std::tolower('<', l) == '<');
    assert(std::tolower('\x8', l) == '\x8');
    assert(std::tolower('A', l) == 'a');
    assert(std::tolower('a', l) == 'a');
    assert(std::tolower('z', l) == 'z');
    assert(std::tolower('3', l) == '3');
    assert(std::tolower('.', l) == '.');
    assert(std::tolower('f', l) == 'f');
    assert(std::tolower('9', l) == '9');
    assert(std::tolower('+', l) == '+');

  return 0;
}
