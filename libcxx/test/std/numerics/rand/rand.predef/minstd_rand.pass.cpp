//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef linear_congruential_engine<uint_fast32_t, 48271, 0, 2147483647>
//                                                                 minstd_rand;

#include <random>
#include <cassert>

int main()
{
    std::minstd_rand e;
    e.discard(9999);
    assert(e() == 399268537u);
}
