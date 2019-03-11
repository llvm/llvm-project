//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// static int xalloc();

#include <ios>
#include <cassert>

int main(int, char**)
{
    int index = std::ios_base::xalloc();
    for (int i = 0; i < 10000; ++i)
        assert(std::ios_base::xalloc() == ++index);

  return 0;
}
