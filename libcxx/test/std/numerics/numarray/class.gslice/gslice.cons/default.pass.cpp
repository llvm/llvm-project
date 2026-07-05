//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// class gslice;

// gslice();

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::gslice gs;
    assert(gs.start() == 0);
    assert(gs.size().size() == 0);
    assert(gs.stride().size() == 0);

  return 0;
}
