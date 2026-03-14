//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isupper (charT c, const locale& loc);

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l;
    assert(!std::isupper(' ', l));
    assert(!std::isupper('<', l));
    assert(!std::isupper('\x8', l));
    assert( std::isupper('A', l));
    assert(!std::isupper('a', l));
    assert(!std::isupper('z', l));
    assert(!std::isupper('3', l));
    assert(!std::isupper('.', l));
    assert(!std::isupper('f', l));
    assert(!std::isupper('9', l));
    assert(!std::isupper('+', l));

  return 0;
}
