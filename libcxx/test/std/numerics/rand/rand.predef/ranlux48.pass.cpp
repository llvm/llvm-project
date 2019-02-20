//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef discard_block_engine<ranlux48_base, 389, 11>                ranlux48;

#include <random>
#include <cassert>

int main(int, char**)
{
    std::ranlux48 e;
    e.discard(9999);
    assert(e() == 249142670248501ull);

  return 0;
}
