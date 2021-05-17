//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N>& reset(size_t pos);

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_C1XX)
#pragma warning(disable: 6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not executed.
#endif

template <std::size_t N>
void test_reset_one() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        for (std::size_t i = 0; i != N; ++i) {
            std::bitset<N> v = cases[c];
            v.reset(i);
            assert(v[i] == false);
        }
    }
}

int main(int, char**) {
    test_reset_one<0>();
    test_reset_one<1>();
    test_reset_one<31>();
    test_reset_one<32>();
    test_reset_one<33>();
    test_reset_one<63>();
    test_reset_one<64>();
    test_reset_one<65>();
    test_reset_one<1000>();

    return 0;
}
