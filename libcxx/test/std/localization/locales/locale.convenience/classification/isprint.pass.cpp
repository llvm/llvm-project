//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isprint (charT c, const locale& loc);

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l;
    assert( std::isprint(' ', l));
    assert( std::isprint('<', l));
    assert(!std::isprint('\x8', l));
    assert( std::isprint('A', l));
    assert( std::isprint('a', l));
    assert( std::isprint('z', l));
    assert( std::isprint('3', l));
    assert( std::isprint('.', l));
    assert( std::isprint('f', l));
    assert( std::isprint('9', l));
    assert( std::isprint('+', l));

  return 0;
}
