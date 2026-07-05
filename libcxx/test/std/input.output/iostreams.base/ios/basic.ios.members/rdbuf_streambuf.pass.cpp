//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// basic_streambuf<charT,traits>* rdbuf(basic_streambuf<charT,traits>* sb);

#include <ios>
#include <streambuf>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::ios ios(0);
    assert(ios.rdbuf() == 0);
    assert(!ios.good());
    std::streambuf* sb = (std::streambuf*)1;
    std::streambuf* sb2 = ios.rdbuf(sb);
    assert(sb2 == 0);
    assert(ios.rdbuf() == sb);
    assert(ios.good());
    sb2 = ios.rdbuf(0);
    assert(sb2 == (std::streambuf*)1);
    assert(ios.rdbuf() == 0);
    assert(ios.bad());

  return 0;
}
