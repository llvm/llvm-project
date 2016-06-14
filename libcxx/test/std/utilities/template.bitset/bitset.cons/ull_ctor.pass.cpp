//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test bitset(unsigned long long val);

#include <bitset>
#include <cassert>
#include <algorithm> // for 'min' and 'max'

#include "test_macros.h"

template <std::size_t N>
void test_val_ctor()
{
    {
        TEST_CONSTEXPR std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
        assert(v.size() == N);
        unsigned M = std::min<std::size_t>(N, 64);
        for (std::size_t i = 0; i < M; ++i)
            assert(v[i] == (i & 1));
        for (std::size_t i = M; i < N; ++i)
            assert(v[i] == false);
    }
#if TEST_STD_VER >= 11
    {
        constexpr std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
        static_assert(v.size() == N, "");
    }
#endif
}

int main()
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
}
