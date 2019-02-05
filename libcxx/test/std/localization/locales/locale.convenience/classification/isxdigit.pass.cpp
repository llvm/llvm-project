//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isxdigit (charT c, const locale& loc);

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l;
    assert(!std::isxdigit(' ', l));
    assert(!std::isxdigit('<', l));
    assert(!std::isxdigit('\x8', l));
    assert( std::isxdigit('A', l));
    assert( std::isxdigit('a', l));
    assert(!std::isxdigit('z', l));
    assert( std::isxdigit('3', l));
    assert(!std::isxdigit('.', l));
    assert( std::isxdigit('f', l));
    assert( std::isxdigit('9', l));
    assert(!std::isxdigit('+', l));

  return 0;
}
