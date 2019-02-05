//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef subtract_with_carry_engine<uint_fast64_t, 48,  5, 12>  ranlux48_base;

#include <random>
#include <cassert>

int main(int, char**)
{
    std::ranlux48_base e;
    e.discard(9999);
    assert(e() == 61839128582725ull);

  return 0;
}
