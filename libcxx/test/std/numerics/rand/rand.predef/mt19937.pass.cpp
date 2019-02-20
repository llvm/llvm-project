//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31,
//                                 0x9908b0df,
//                                 11, 0xffffffff,
//                                 7,  0x9d2c5680,
//                                 15, 0xefc60000,
//                                 18, 1812433253>                      mt19937;

#include <random>
#include <cassert>

int main(int, char**)
{
    std::mt19937 e;
    e.discard(9999);
    assert(e() == 4123659995u);

  return 0;
}
