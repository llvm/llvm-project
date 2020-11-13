//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset(unsigned long long val);

#include <bitset>
#include <cassert>
#include <algorithm> // for 'min' and 'max'
#include <cstddef>

#include "test_macros.h"

#if defined(TEST_COMPILER_C1XX)
#pragma warning(disable: 6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not executed.
#endif

template <std::size_t N>
void test_val_ctor()
{
    {
        TEST_CONSTEXPR std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
        assert(v.size() == N);
        std::size_t M = std::min<std::size_t>(v.size(), 64);
        for (std::size_t i = 0; i < M; ++i)
            assert(v[i] == ((i & 1) != 0));
        for (std::size_t i = M; i < v.size(); ++i)
            assert(v[i] == false);
    }
#if TEST_STD_VER >= 11
    {
        constexpr std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
        static_assert(v.size() == N, "");
    }
#endif
}

int main(int, char**)
{
    test_val_ctor<0>();
    test_val_ctor<1>();
    test_val_ctor<31>();
    test_val_ctor<32>();
    test_val_ctor<33>();
    test_val_ctor<63>();
    test_val_ctor<64>();
    test_val_ctor<65>();
    test_val_ctor<1000>();

  return 0;
}
