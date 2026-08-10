//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isdigit (charT c, const locale& loc);

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l;
    assert(!std::isdigit(' ', l));
    assert(!std::isdigit('<', l));
    assert(!std::isdigit('\x8', l));
    assert(!std::isdigit('A', l));
    assert(!std::isdigit('a', l));
    assert(!std::isdigit('z', l));
    assert( std::isdigit('3', l));
    assert(!std::isdigit('.', l));
    assert(!std::isdigit('f', l));
    assert( std::isdigit('9', l));
    assert(!std::isdigit('+', l));

  return 0;
}
