//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// static const iostate badbit;
// static const iostate eofbit;
// static const iostate failbit;
// static const iostate goodbit = 0;

#include <ios>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    assert(std::ios_base::badbit);
    assert(std::ios_base::eofbit);
    assert(std::ios_base::failbit);

    assert
    (
        ( std::ios_base::badbit
        & std::ios_base::eofbit
        & std::ios_base::failbit) == 0
    );

    assert(std::ios_base::goodbit == 0);

  return 0;
}
