//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// streamsize width(streamsize wide);

#include <ios>
#include <cassert>

#include "test_macros.h"

class test
    : public std::ios
{
public:
    test()
    {
        init(0);
    }
};

int main(int, char**)
{
    test t;
    assert(t.width() == 0);
    std::streamsize w = t.width(4);
    assert(w == 0);
    assert(t.width() == 4);

  return 0;
}
