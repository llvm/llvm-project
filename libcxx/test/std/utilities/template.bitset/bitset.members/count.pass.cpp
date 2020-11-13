//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test size_t count() const;

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
void test_count() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        const std::bitset<N> v = cases[c];
        std::size_t c1 = v.count();
        std::size_t c2 = 0;
        for (std::size_t i = 0; i < v.size(); ++i)
            if (v[i])
                ++c2;
        assert(c1 == c2);
    }
}

int main(int, char**) {
    test_count<0>();
    test_count<1>();
    test_count<31>();
    test_count<32>();
    test_count<33>();
    test_count<63>();
    test_count<64>();
    test_count<65>();
    test_count<1000>();

    return 0;
}
