//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// ios_base& noshowbase(ios_base& str);

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf : public std::streambuf {};

int main(int, char**)
{
    testbuf sb;
    std::ios ios(&sb);
    std::showbase(ios);
    std::ios_base& r = std::noshowbase(ios);
    assert(&r == &ios);
    assert(!(ios.flags() & std::ios::showbase));

  return 0;
}
