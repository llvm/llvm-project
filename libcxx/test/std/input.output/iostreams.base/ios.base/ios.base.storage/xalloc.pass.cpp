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

int main()
{
    assert(std::ios_base::xalloc() == 0);
    assert(std::ios_base::xalloc() == 1);
    assert(std::ios_base::xalloc() == 2);
    assert(std::ios_base::xalloc() == 3);
    assert(std::ios_base::xalloc() == 4);
}
