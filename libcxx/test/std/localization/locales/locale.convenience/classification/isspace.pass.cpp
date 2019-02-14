//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isspace (charT c, const locale& loc);

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l;
    assert( std::isspace(' ', l));
    assert(!std::isspace('<', l));
    assert(!std::isspace('\x8', l));
    assert(!std::isspace('A', l));
    assert(!std::isspace('a', l));
    assert(!std::isspace('z', l));
    assert(!std::isspace('3', l));
    assert(!std::isspace('.', l));
    assert(!std::isspace('f', l));
    assert(!std::isspace('9', l));
    assert(!std::isspace('+', l));

  return 0;
}
