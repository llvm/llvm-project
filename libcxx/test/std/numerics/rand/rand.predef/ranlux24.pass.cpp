//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// typedef discard_block_engine<ranlux24_base, 223, 23>                ranlux24;

#include <random>
#include <cassert>

int main()
{
    std::ranlux24 e;
    e.discard(9999);
    assert(e() == 9901578u);
}
