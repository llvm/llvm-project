//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool ispunct (charT c, const locale& loc);

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l;
    assert(!std::ispunct(' ', l));
    assert( std::ispunct('<', l));
    assert(!std::ispunct('\x8', l));
    assert(!std::ispunct('A', l));
    assert(!std::ispunct('a', l));
    assert(!std::ispunct('z', l));
    assert(!std::ispunct('3', l));
    assert( std::ispunct('.', l));
    assert(!std::ispunct('f', l));
    assert(!std::ispunct('9', l));
    assert( std::ispunct('+', l));

  return 0;
}
