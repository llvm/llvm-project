//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef mersenne_twister_engine<uint_fast64_t, 64, 312, 156, 31,
//                                 0xb5026f5aa96619e9,
//                                 29, 0x5555555555555555,
//                                 17, 0x71d67fffeda60000,
//                                 37, 0xfff7eee000000000,
//                                 43, 6364136223846793005>          mt19937_64;

#include <random>
#include <cassert>

int main()
{
    std::mt19937_64 e;
    e.discard(9999);
    assert(e() == 9981545732273789042ull);
}
