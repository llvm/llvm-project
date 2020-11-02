//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N>& set();

#include <bitset>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

template <std::size_t N>
void test_set_all() {
    std::bitset<N> v;
    v.set();
    for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i]);
}

int main(int, char**) {
    test_set_all<0>();
    test_set_all<1>();
    test_set_all<31>();
    test_set_all<32>();
    test_set_all<33>();
    test_set_all<63>();
    test_set_all<64>();
    test_set_all<65>();
    test_set_all<1000>();

    return 0;
}
