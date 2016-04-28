//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test bitset<N>& reset();

#include <bitset>
#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wtautological-compare"
#endif

template <std::size_t N>
void test_reset_all()
{
    std::bitset<N> v;
    v.set();
    v.reset();
    for (std::size_t i = 0; i < N; ++i)
        assert(!v[i]);
}

int main()
{
    test_reset_all<0>();
    test_reset_all<1>();
    test_reset_all<31>();
    test_reset_all<32>();
    test_reset_all<33>();
    test_reset_all<63>();
    test_reset_all<64>();
    test_reset_all<65>();
    test_reset_all<1000>();
}
