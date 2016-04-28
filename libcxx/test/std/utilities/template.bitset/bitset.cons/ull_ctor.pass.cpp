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

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wtautological-compare"
#endif

template <std::size_t N>
void test_val_ctor()
{
    {
    _LIBCPP_CONSTEXPR std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
    assert(v.size() == N);
    unsigned M = std::min<std::size_t>(N, 64);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (i & 1));
    for (std::size_t i = M; i < N; ++i)
        assert(v[i] == false);
    }
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
