//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N>& set(size_t pos, bool val = true);

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
void test_set_one() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        std::bitset<N> v = cases[c];
        if (v.size() > 0) {
            std::size_t middle = v.size() / 2;
            v.set(middle);
            assert(v[middle] == true);
            v.set(middle, false);
            assert(v[middle] == false);
        }
    }
}

int main(int, char**) {
    test_set_one<0>();
    test_set_one<1>();
    test_set_one<31>();
    test_set_one<32>();
    test_set_one<33>();
    test_set_one<63>();
    test_set_one<64>();
    test_set_one<65>();
    test_set_one<1000>();

    return 0;
}
