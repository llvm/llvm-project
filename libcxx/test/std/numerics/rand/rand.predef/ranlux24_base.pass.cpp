//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef subtract_with_carry_engine<uint_fast32_t, 24, 10, 24>  ranlux24_base;

#include <random>
#include <cassert>

int main(int, char**)
{
    std::ranlux24_base e;
    e.discard(9999);
    assert(e() == 7937952u);

  return 0;
}
