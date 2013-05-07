//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test:

// bool operator==(const bitset<N>& rhs) const;
// bool operator!=(const bitset<N>& rhs) const;

#include <bitset>
#include <cstdlib>
#include <cassert>

#pragma clang diagnostic ignored "-Wtautological-compare"

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
void test_equality()
{
    const std::bitset<N> v1 = make_bitset<N>();
    std::bitset<N> v2 = v1;
    assert(v1 == v2);
    if (N > 0)
    {
        v2[N/2].flip();
        assert(v1 != v2);
    }
}

int main()
{
    test_equality<0>();
    test_equality<1>();
    test_equality<31>();
    test_equality<32>();
    test_equality<33>();
    test_equality<63>();
    test_equality<64>();
    test_equality<65>();
    test_equality<1000>();
}
