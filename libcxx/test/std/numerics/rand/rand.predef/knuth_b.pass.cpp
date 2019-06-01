//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef shuffle_order_engine<minstd_rand0, 256>                      knuth_b;

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::knuth_b e;
    e.discard(9999);
    assert(e() == 1112339016u);

  return 0;
}
