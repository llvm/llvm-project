//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isgraph (charT c, const locale& loc);

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l;
    assert(!std::isgraph(' ', l));
    assert( std::isgraph('<', l));
    assert(!std::isgraph('\x8', l));
    assert( std::isgraph('A', l));
    assert( std::isgraph('a', l));
    assert( std::isgraph('z', l));
    assert( std::isgraph('3', l));
    assert( std::isgraph('.', l));
    assert( std::isgraph('f', l));
    assert( std::isgraph('9', l));
    assert( std::isgraph('+', l));

  return 0;
}
