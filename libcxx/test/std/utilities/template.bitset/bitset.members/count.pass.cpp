//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test size_t count() const;

#include <bitset>
#include <cstdlib>
#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wtautological-compare"
#endif

template <std::size_t N>
std::bitset<N>
make_bitset()
{
    std::bitset<N> v;
    for (std::size_t i = 0; i < N; ++i)
        v[i] = static_cast<bool>(std::rand() & 1);
    return v;
}

template <std::size_t N>
void test_count()
{
    const std::bitset<N> v = make_bitset<N>();
    std::size_t c1 = v.count();
    std::size_t c2 = 0;
    for (std::size_t i = 0; i < N; ++i)
        if (v[i])
            ++c2;
    assert(c1 == c2);
}

int main()
{
    test_count<0>();
    test_count<1>();
    test_count<31>();
    test_count<32>();
    test_count<33>();
    test_count<63>();
    test_count<64>();
    test_count<65>();
    test_count<1000>();
}
