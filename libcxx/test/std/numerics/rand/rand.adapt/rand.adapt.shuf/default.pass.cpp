//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t k>
// class shuffle_order_engine

// explicit shuffle_order_engine();

#include <random>
#include <cassert>

void
test1()
{
    std::knuth_b e1;
    std::knuth_b e2(std::minstd_rand0::default_seed);
    assert(e1 == e2);
    assert(e1() == 152607844u);
}

int main()
{
    test1();
}
